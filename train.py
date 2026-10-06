import argparse
import torch
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from torch.cuda import init

# custom
from dassl.data.datasets import VisDA17
from dassl.data.datasets import OfficeHome
from datasets.officehome_multitarget import OfficeHomeMultiTarget
from datasets.aerial_multitarget import AerialMultiTarget
from datasets.aerial_single_lumped import AerialSingleLumped

import trainers.dapl


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables for DAPL.
    """
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.TRAINER.DAPL = CN()
    cfg.DATASET.SHARED_CLASSES = [] 
    cfg.TRAINER.DAPL.N_DMX = 16          # number of DSC tokens
    cfg.TRAINER.DAPL.N_CTX = 16          # number of context vectors
    cfg.TRAINER.DAPL.CSC = False          # class-specific context
    cfg.TRAINER.DAPL.PREC = "fp16"        # fp16, fp32, amp
    cfg.TRAINER.DAPL.T = 1.0
    cfg.TRAINER.DAPL.TAU = 0.5
    cfg.TRAINER.DAPL.U = 1.0
    cfg.DATASET.SHARED_CLASSES = []
    cfg.DATASET.LUMP_DOMAINS = []      # <-- NEW, required for the family restriction
    # Visual prompts: learnable per-domain residual vectors added to the
    # global image embedding after the image encoder.
    cfg.TRAINER.DAPL.USE_VISUAL_PROMPTS = True
    cfg.TRAINER.DAPL.VISUAL_PROMPT_LR = 3e-4
    cfg.TRAINER.DAPL.VP_MAX_NORM = 0.1

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    print(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def setup_multi_target_trainer(trainer, cfg):
    """Setup trainer for multiple target domains using existing data manager."""
    from dassl.data.data_manager import DataManager

    target_loaders = []

    for target_domain in cfg.DATASET.TARGET_DOMAINS:
        print(f"Creating data loader for target domain: {target_domain}")

        # Create a temporary config for each target domain
        temp_cfg = cfg.clone()
        temp_cfg.defrost()
        temp_cfg.DATASET.TARGET_DOMAINS = [target_domain]
        temp_cfg.freeze()

        # Create a new data manager for this target domain
        temp_dm = DataManager(temp_cfg)

        # Get the train_u loader from this data manager
        target_loader = temp_dm.train_loader_u
        target_loaders.append(target_loader)

        print(f"Created target loader for domain: {target_domain} "
              f"with {len(temp_dm.dataset.train_u)} samples")

    # Replace the single train_loader_u with list of loaders
    trainer.train_loader_u_list     = target_loaders
    trainer.train_loader_u_original = trainer.train_loader_u

    return trainer


def override_dataset_for_multitarget(cfg):
    """Override dataset class when multiple target domains are specified."""
    if len(cfg.DATASET.TARGET_DOMAINS) > 1 and cfg.DATASET.NAME == "OfficeHome":
        cfg.defrost()
        cfg.DATASET.NAME = "OfficeHomeMultiTarget"
        cfg.freeze()
        print(f"Switched to OfficeHomeMultiTarget for multi-target adaptation")
    return cfg


def main(args):
    cfg = setup_cfg(args)

    # Override dataset for multi-target if needed
    cfg = override_dataset_for_multitarget(cfg)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    print("Collecting env info ...")
    try:
        print("** System info **\n{}\n".format(collect_env_info()))
    except Exception as e:
        print(f"Warning: Could not collect environment info: {e}")
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA version:", torch.version.cuda)
            print("GPU count:", torch.cuda.device_count())

    trainer = build_trainer(cfg)

    # Debug data loading
    try:
        print(f"\n=== Data Loading Debug ===")
        print(f"Dataset: {cfg.DATASET.NAME}")
        print(f"Source domains: {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Target domains: {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Dataset root: {cfg.DATASET.ROOT}")

        dataset_path = os.path.join(cfg.DATASET.ROOT, "office_home")
        print(f"Looking for dataset at: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")

        if os.path.exists(dataset_path):
            domains = os.listdir(dataset_path)
            print(f"Available domains: {domains}")

            for domain in cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS:
                domain_path = os.path.join(dataset_path, domain)
                print(f"Domain '{domain}' exists: {os.path.exists(domain_path)}")
                if os.path.exists(domain_path):
                    classes = os.listdir(domain_path)
                    print(f"  Classes in {domain}: {len(classes)}")

        print(f"Train_x loader length: "
              f"{len(trainer.train_loader_x) if hasattr(trainer, 'train_loader_x') else 'Not available'}")
        print(f"Train_u loader length: "
              f"{len(trainer.train_loader_u) if hasattr(trainer, 'train_loader_u') else 'Not available'}")
        print(f"Test loader length: "
              f"{len(trainer.test_loader) if hasattr(trainer, 'test_loader') else 'Not available'}")
        print(f"=== Debug Complete ===\n")

    except Exception as e:
        print(f"Debug error: {e}")

    # Setup multi-target support
    if len(cfg.DATASET.TARGET_DOMAINS) > 1:
        print(f"Multi-target mode: {len(cfg.DATASET.TARGET_DOMAINS)} domains: "
              f"{cfg.DATASET.TARGET_DOMAINS}")
        trainer = setup_multi_target_trainer(trainer, cfg)
        trainer.multi_target_mode = True
    else:
        trainer.multi_target_mode = False

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="",
                        help="output directory")
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+",
                        help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+",
                        help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+",
                        help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="",
                        help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="path to config file for dataset setup")
    parser.add_argument("--trainer", type=str, default="",
                        help="name of trainer")
    parser.add_argument("--backbone", type=str, default="",
                        help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="",
                        help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int,
                        help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true",
                        help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
