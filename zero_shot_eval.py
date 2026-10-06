"""
Zero-shot CLIP benchmark on the cross-dataset aerial closed set.

This gives the honest "no adaptation" baseline that your trained multi-target
numbers must be compared against. It reuses:
  * the SAME CLIP backbone + loader as the trainer (load_clip_to_cpu)
  * the SAME AerialMultiTarget dataset / closed-set label space
  * the SAME hand-crafted template DAPL uses for its naive head:
        "a photo of a [class]."

It does NOT use any learned text or visual prompts — it is pure zero-shot
CLIP, evaluated per target domain and averaged, mirroring how the trainer
reports per-domain + mean accuracy.

Usage (from the project root, same env as training):

  python zero_shot_eval.py \
      --root D:/VIP/Multidomain \
      --dataset-config-file configs/datasets/aerial.yaml \
      --config-file configs/trainers/DAPL/ep25-32.yaml \
      --source-domains AID \
      --target-domains CLRS NWPU UCM
"""

import argparse
import torch
import torch.nn.functional as F

from dassl.config import get_cfg_default
from dassl.data import DatasetWrapper
from dassl.data.transforms import build_transform
from torch.utils.data import DataLoader

from clip import clip

# Register datasets (triggers DATASET_REGISTRY.register()).
from dassl.data.datasets import build_dataset
import datasets.officehome_multitarget  # noqa: F401
import datasets.aerial_multitarget      # noqa: F401

# Reuse the trainer's exact CLIP loader so the backbone matches 1:1.
from trainers.dapl import load_clip_to_cpu


def extend_cfg(cfg):
    """Mirror the config keys train.py adds, so the YAMLs merge cleanly."""
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.DATASET.SHARED_CLASSES = []
    cfg.TRAINER.DAPL = CN()
    cfg.TRAINER.DAPL.N_DMX = 16
    cfg.TRAINER.DAPL.N_CTX = 16
    cfg.TRAINER.DAPL.CSC = False
    cfg.TRAINER.DAPL.PREC = "fp16"
    cfg.TRAINER.DAPL.T = 1.0
    cfg.TRAINER.DAPL.TAU = 0.5
    cfg.TRAINER.DAPL.U = 1.0
    cfg.TRAINER.DAPL.USE_VISUAL_PROMPTS = True
    cfg.TRAINER.DAPL.VISUAL_PROMPT_LR = 3e-4
    cfg.TRAINER.DAPL.VP_MAX_NORM = 0.1


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    cfg.freeze()
    return cfg


@torch.no_grad()
def build_zeroshot_text_features(clip_model, classnames, device):
    """Encode 'a photo of a [class].' for each class -> normalised features."""
    classnames = [c.replace("_", " ") for c in classnames]
    prompts = [f"a photo of a {c}." for c in classnames]
    tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    text_features = clip_model.encode_text(tokens)
    text_features = F.normalize(text_features, dim=-1)
    return text_features  # [C, D]


@torch.no_grad()
def evaluate_domain(clip_model, text_features, loader, device):
    correct = total = 0
    logit_scale = clip_model.logit_scale.exp()
    for batch in loader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def make_test_loader(cfg, single_target):
    """Build a test loader for one target domain, matching the trainer."""
    temp = cfg.clone()
    temp.defrost()
    temp.DATASET.TARGET_DOMAINS = [single_target]
    temp.freeze()

    dataset = build_dataset(temp)
    tfm_test = build_transform(temp, is_train=False)
    loader = DataLoader(
        DatasetWrapper(temp, dataset.test, transform=tfm_test, is_train=False),
        batch_size=temp.DATALOADER.TEST.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=temp.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def main(args):
    cfg = setup_cfg(args)
    device = "cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu"

    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME}) ...")
    clip_model = load_clip_to_cpu(cfg).to(device).eval()

    target_domains = list(cfg.DATASET.TARGET_DOMAINS)
    print(f"Source: {cfg.DATASET.SOURCE_DOMAINS} | Targets: {target_domains}")

    accs = []
    classnames_ref = None
    for tdom in target_domains:
        dataset, loader = make_test_loader(cfg, tdom)
        classnames = dataset.classnames
        classnames_ref = classnames
        text_features = build_zeroshot_text_features(clip_model, classnames, device)
        acc = evaluate_domain(clip_model, text_features, loader, device)
        accs.append(acc)
        print(f"  [Zero-shot] Domain [{tdom}]: accuracy = {acc:.2f}%")

    mean_acc = sum(accs) / max(len(accs), 1)
    print(f"\nClasses ({len(classnames_ref)}): {classnames_ref}")
    print(f"[Zero-shot CLIP] Mean accuracy across "
          f"{len(target_domains)} target domains: {mean_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
