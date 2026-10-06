import os.path as osp
import os
import datetime
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DAPL.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        domainnames = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS
        domainnames = [
            ", a {} image.".format(domain) for domain in domainnames
        ]
        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(
            cfg.DATASET.TARGET_DOMAINS)  # number of domains

        self.n_target_domains = len(cfg.DATASET.TARGET_DOMAINS)

        n_dmx = cfg.TRAINER.DAPL.N_DMX  # number of domain context
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = "a photo of a".replace("_", " ")

        if cfg.TRAINER.DAPL.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        print("ctx vectors size: ".format(ctx_vectors.size()))
        prompt_prefix = " ".join(["X"] * n)

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]

        prompts = [
            prompt_prefix + " " + name + " " + domain + "."
            for domain in domainnames for name in classnames
        ]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)
            naive_embedding = clip_model.token_embedding(
                naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        tokenized_prompts = torch.cat(
            [tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:,
                                                       1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.DAPL.CSC
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.to(
            torch.device("cuda"))

    @autocast()
    def forward(self):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors  # dm 16 512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1,
                                              -1)  # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1,
                                          -1)  # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx],
                           dim=2).reshape(self.n_cls * self.n_dm,
                                          self.n_ctx + self.n_dmx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # naive
        neb = self.naive_embedding

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctxdmx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat([prompts, neb], dim=0)

        return prompts


# ---------------------------------------------------------------------------
# Visual Prompt Learner
# ---------------------------------------------------------------------------

class VisualPromptLearner(nn.Module):
    """
    Domain-specific visual prompts added to the global image embedding
    *after* the image encoder (compatible with RN50 — no ViT layer injection).

    Parameters
    ----------
    visual_prompts : nn.Parameter  [n_domains, embed_dim]
        One learnable residual vector per domain.  Initialised to **zero**
        so the model starts as a plain DAPL run and learns domain-specific
        corrections from there.

    Training:  domain identity is known → add visual_prompts[domain_idx].
    Test time: domain identity is unknown → the DAPL trainer uses the same
               confidence-weighted attention mechanism as for the text heads
               (see DAPL.test()).
    """

    def __init__(self, cfg, clip_model):
        super().__init__()
        n_domains  = (len(cfg.DATASET.SOURCE_DOMAINS) +
                      len(cfg.DATASET.TARGET_DOMAINS))
        embed_dim  = clip_model.visual.output_dim   # 1024 for RN50
        dtype      = clip_model.dtype

        # Zero-init → no-op at epoch 0; learns residuals from there
        visual_prompts = torch.zeros(n_domains, embed_dim, dtype=dtype)
        self.visual_prompts = nn.Parameter(visual_prompts)

        self.n_domains = n_domains
        self.embed_dim = embed_dim

        print(f"VisualPromptLearner  |  "
              f"n_domains={n_domains}  embed_dim={embed_dim}  dtype={dtype}")

    def forward(self, image_features: torch.Tensor,
                domain_idx: int) -> torch.Tensor:
        """
        Apply the domain prompt for a known domain during training.

        Args:
            image_features: [B, embed_dim]
            domain_idx:     scalar int — which domain's prompt to add

        Returns:
            [B, embed_dim]  (same shape, with prompt residual added)
        """
        vp = self.visual_prompts[domain_idx]          # [D]
        return image_features + vp.unsqueeze(0)       # [B, D]  (broadcast)

    def forward_weighted(self, image_features: torch.Tensor,
                         weights: torch.Tensor,
                         domain_indices: list) -> torch.Tensor:
        """
        Attention-weighted combination of domain prompts — domain-agnostic
        test-time variant.

        Args:
            image_features: [B, embed_dim]  — raw (un-prompted) features
            weights:        [B, n_domains]  — soft weights (sum to 1 per image)
            domain_indices: list of int, length n_domains

        Returns:
            [B, embed_dim]  — weighted-sum-prompted features
        """
        vps = self.visual_prompts[domain_indices]     # [n_domains, D]
        # [B, n_domains, D] × [B, n_domains, 1] → [B, D]
        weighted_vp = (weights.unsqueeze(-1) * vps.unsqueeze(0)).sum(dim=1)
        return image_features + weighted_vp


# ---------------------------------------------------------------------------
# Wrapper so both learnable modules share one optimizer
# ---------------------------------------------------------------------------

class AllLearnable(nn.Module):
    """
    Thin wrapper that holds PromptLearner + VisualPromptLearner together so
    they can be passed as a single module to build_optimizer / register_model.
    The state_dict keys are 'prompt_learner.*' and 'visual_prompt_learner.*'.
    """

    def __init__(self, prompt_learner: nn.Module,
                 visual_prompt_learner: nn.Module):
        super().__init__()
        self.prompt_learner       = prompt_learner
        self.visual_prompt_learner = visual_prompt_learner


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner    = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder     = clip_model.visual
        self.text_encoder      = TextEncoder(clip_model)
        self.logit_scale       = clip_model.logit_scale
        self.dtype             = clip_model.dtype
        # visual_prompt_learner is attached after construction in build_model()
        self.visual_prompt_learner: VisualPromptLearner | None = None

    @autocast()
    def forward(self, image: torch.Tensor,
                domain_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            image:      [B, C, H, W]
            domain_idx: if not None, apply visual_prompts[domain_idx] to the
                        image features before computing logits (training only).

        Returns:
            logits [B, n_dm*n_cls + n_cls]
        """
        image_features = self.image_encoder(image.type(self.dtype))

        # ── Visual prompt (training, domain known) ──────────────────────────
        if (domain_idx is not None and
                self.visual_prompt_learner is not None):
            image_features = self.visual_prompt_learner(
                image_features, domain_idx)

        prompts            = self.prompt_learner()
        tokenized_prompts  = self.tokenized_prompts
        text_features      = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features  = text_features  / text_features .norm(
            dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits      = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class DAPL(TrainerXU):
    """Domain Adaptation via Prompt Learning (DAPL) — extended with
    domain-specific visual prompts.

    Visual prompts are learnable residual vectors added to the global image
    embedding *post-encoder* (compatible with RN50).  At test time the domain
    identity is unknown, so we use the same confidence-weighted attention
    mechanism as for the text heads to combine domain-specific logits.

    Reference: https://arxiv.org/abs/2202.06687
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DAPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg        = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DAPL.PREC in ("fp32", "amp"):
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # ── Visual prompt learner ────────────────────────────────────────────
        use_vp = getattr(cfg.TRAINER.DAPL, "USE_VISUAL_PROMPTS", True)
        if use_vp:
            vpl = VisualPromptLearner(cfg, clip_model)
            self.model.visual_prompt_learner = vpl
        # also expose via self for convenience in forward_backward
        self.use_visual_prompts = use_vp

        # plus one for pseudo label per target domain
        self.n_dm           = (self.model.prompt_learner.n_dm +
                               self.model.prompt_learner.n_target_domains)
        self.n_cls          = self.model.prompt_learner.n_cls
        self.n_target_domains = self.model.prompt_learner.n_target_domains
        self.source_count   = len(cfg.DATASET.SOURCE_DOMAINS)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "visual_prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # ── Optimizer: covers both PromptLearner + VisualPromptLearner ───────
        # AllLearnable wraps them so build_optimizer sees one module whose
        # state_dict has keys  "prompt_learner.*" and "visual_prompt_learner.*"
        if use_vp:
            self._all_learnable = AllLearnable(
                self.model.prompt_learner,
                self.model.visual_prompt_learner)
        else:
            # backward-compatible: only prompt_learner in optimizer
            self._all_learnable = AllLearnable(
                self.model.prompt_learner,
                nn.Module())   # empty module, no parameters

        if self.use_visual_prompts:
            vp_lr = getattr(cfg.TRAINER.DAPL, "VISUAL_PROMPT_LR", cfg.OPTIM.LR * 0.1)
            param_groups = [
                {"params": list(self.model.prompt_learner.parameters()),
                "lr": cfg.OPTIM.LR},
                {"params": list(self.model.visual_prompt_learner.parameters()),
                "lr": vp_lr},
            ]
            self.optim = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True,
            )
        else:
            self.optim = build_optimizer(self._all_learnable, cfg.OPTIM)

        total_steps = cfg.OPTIM.MAX_EPOCH * self.num_batches
        warmup_steps = cfg.OPTIM.WARMUP_EPOCH * self.num_batches

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup_sched = LinearLR(
            self.optim,
            start_factor=cfg.OPTIM.WARMUP_MIN_LR / cfg.OPTIM.LR,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_sched = CosineAnnealingLR(
            self.optim,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.OPTIM.WARMUP_MIN_LR,
        )
        self.sched = SequentialLR(
            self.optim,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )
        self.register_model("prompt_learner", self._all_learnable,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DAPL.PREC == "amp" else None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names      = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = torch.load(model_path, map_location="cpu",
                                    weights_only=False)
            state_dict = checkpoint["state_dict"]
            epoch_ckpt = checkpoint["epoch"]

            # Drop buffers that are recomputed from current class names
            for key in ("token_prefix", "token_suffix",
                        "prompt_learner.token_prefix",
                        "prompt_learner.token_suffix"):
                state_dict.pop(key, None)

            print(f"Loading weights to {name} from \"{model_path}\" "
                  f"(epoch = {epoch_ckpt})")
            # strict=False: tolerates missing visual_prompt_learner in old
            # checkpoints and tolerates missing keys in new-→-old direction
            self._models[name].load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses     = MetricMeter()
        batch_time = AverageMeter()
        data_time  = AverageMeter()

        len_train_loader_x = len(self.train_loader_x)

        # ── Determine num_batches and build iterators ────────────────────────
        if hasattr(self, 'train_loader_u_list'):
            len_train_loader_u_list = [len(ldr) for ldr in self.train_loader_u_list]

            if self.cfg.TRAIN.COUNT_ITER == "train_x":
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == "train_u":
                self.num_batches = max(len_train_loader_u_list)
            elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
                self.num_batches = min(len_train_loader_x,
                                    min(len_train_loader_u_list))
            else:
                raise ValueError

            train_loader_x_iter      = iter(self.train_loader_x)
            train_loader_u_iter_list = [iter(ldr) for ldr in
                                        self.train_loader_u_list]
        else:
            len_train_loader_u = len(self.train_loader_u)

            if self.cfg.TRAIN.COUNT_ITER == "train_x":
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == "train_u":
                self.num_batches = len_train_loader_u
            elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
                self.num_batches = min(len_train_loader_x, len_train_loader_u)
            else:
                raise ValueError

            train_loader_x_iter = iter(self.train_loader_x)
            train_loader_u_iter = iter(self.train_loader_u)

        # ── Progress bar ─────────────────────────────────────────────────────
        import sys
        pbar = tqdm(
            range(self.num_batches),
            desc=f"Epoch [{self.epoch+1}/{self.max_epoch}]",
            ncols=120,
            leave=True,
            disable=not sys.stderr.isatty(),  # disables dynamic bar in SLURM, keeps it locally
        )

        end = time.time()
        for self.batch_idx in pbar:

            # ── Fetch source batch ───────────────────────────────────────────
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            # ── Fetch target batch(es) and run forward-backward ─────────────
            if hasattr(self, 'train_loader_u_list'):
                batch_u_list = []
                for i, it in enumerate(train_loader_u_iter_list):
                    try:
                        batch_u = next(it)
                    except StopIteration:
                        train_loader_u_iter_list[i] = iter(
                            self.train_loader_u_list[i])
                        batch_u = next(train_loader_u_iter_list[i])
                    batch_u_list.append(batch_u)

                data_time.update(time.time() - end)
                loss_summary = self.forward_backward(batch_x, batch_u_list)

            else:
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.train_loader_u)
                    batch_u = next(train_loader_u_iter)

                data_time.update(time.time() - end)
                loss_summary = self.forward_backward(batch_x, batch_u)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # ── Update progress bar ──────────────────────────────────────────
            pbar.set_postfix({
                "loss":   f"{loss_summary['loss']:.3f}",
                "loss_x": f"{loss_summary['loss_x']:.3f}",
                "loss_u": f"{loss_summary['loss_u']:.3f}",
                "acc_x":  f"{loss_summary['acc_x']:.1f}%",
                "lr":     f"{self.get_current_lr():.2e}",
            })

            # ── Periodic log line (goes to log file via stdout) ──────────────
            if ((self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or
                    self.num_batches < self.cfg.TRAIN.PRINT_FREQ):
                nb_remain  = (self.num_batches - self.batch_idx - 1)
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(
                    seconds=int(batch_time.avg * nb_remain)))
                tqdm.write(
                    f"epoch [{self.epoch+1}/{self.max_epoch}]"
                    f"[{self.batch_idx+1}/{self.num_batches}]  "
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})  "
                    f"eta {eta}  "
                    f"{losses}  "
                    f"lr {self.get_current_lr():.6e}"
                )

            # ── Tensorboard ──────────────────────────────────────────────────
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward_backward(self, batch_x, batch_u):
        if isinstance(batch_u, list):
            return self.forward_backward_multi_target(batch_x, batch_u)
        else:
            return self.forward_backward_single_target(batch_x, batch_u)

    def forward_backward_single_target(self, batch_x, batch_u):
        """Single target forward_backward — domain indices are fixed:
        source=0, target=source_count (=1 for OfficeHome single-source)."""
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.DAPL.PREC

        source_idx = 0
        target_idx = self.source_count  # = 1

        if prec == "amp":
            with autocast():
                output_x = self.model(image_x, domain_idx=source_idx)
                output_u = self.model(image_u, domain_idx=target_idx)

                # only clip annotation
                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1, self.n_cls) /
                    self.cfg.TRAINER.DAPL.T,
                    dim=-1)

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()

                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
                if mask.sum() > 0:
                    loss_u = (F.cross_entropy(
                        output_u[:, self.n_cls:2 * self.n_cls],
                        label_p,
                        reduction="none") * mask).sum() / mask.sum()
                else:
                    loss_u = torch.tensor(0.0, device=self.device)
                if self.use_visual_prompts and self.model.visual_prompt_learner is not None:
                    vp_reg = self.model.visual_prompt_learner.visual_prompts.pow(2).mean()
                else:
                    vp_reg = torch.tensor(0.0, device=self.device)
                ramp = min(1.0, self.epoch / 10.0)
                loss = loss_x + self.cfg.TRAINER.DAPL.U * ramp * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            if self.use_visual_prompts and self.model.visual_prompt_learner is not None:
                with torch.no_grad():
                    vp = self.model.visual_prompt_learner.visual_prompts
                    max_norm = getattr(self.cfg.TRAINER.DAPL, "VP_MAX_NORM", 0.1)
                    norms = vp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    vp.mul_((norms.clamp(max=max_norm) / norms))

        loss_summary = {
            "loss":   loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x":  compute_accuracy(
                output_x[:, :self.n_cls], label)[0].item(),
        }

        self.update_lr()
        return loss_summary

    def forward_backward_multi_target(self, batch_x, batch_u_list):
        """Multi-target forward_backward — each target gets its own
        domain index so visual prompts are domain-specific."""
        image_x = batch_x["img"].to(self.device)
        label   = batch_x["label"].to(self.device)

        prec = self.cfg.TRAINER.DAPL.PREC

        if prec == "amp":
            with autocast():
                # Source domain (always index 0)
                output_x = self.model(image_x, domain_idx=0)

                loss_u_total = 0.0
                mask_count   = 0

                for target_idx, batch_u in enumerate(batch_u_list):
                    image_u    = batch_u["img"].to(self.device)
                    domain_idx = self.source_count + target_idx

                    # Pass domain index so the correct visual prompt is used
                    output_u = self.model(image_u, domain_idx=domain_idx)

                    # Pseudo labels from the naive CLIP slice (last n_cls)
                    total_domains   = (self.source_count +
                                       len(self.cfg.DATASET.TARGET_DOMAINS))
                    clip_start_idx  = self.n_cls * total_domains
                    pseudo_label    = torch.softmax(
                        output_u[:, clip_start_idx:clip_start_idx + self.n_cls]
                        / self.cfg.TRAINER.DAPL.T,
                        dim=-1)

                    max_probs, label_p = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()

                    if mask.sum() > 0:
                        t_start = self.n_cls * domain_idx
                        loss_u_target = (F.cross_entropy(
                            output_u[:, t_start:t_start + self.n_cls],
                            label_p,
                            reduction="none") * mask).sum() / mask.sum()
                        loss_u_total += loss_u_target
                        mask_count   += 1

                if mask_count > 0:
                    loss_u = loss_u_total / mask_count
                else:
                    loss_u = torch.tensor(0.0, device=self.device)

                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
                loss   = loss_x + self.cfg.TRAINER.DAPL.U * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        loss_summary = {
            "loss":   loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x":  compute_accuracy(
                output_x[:, :self.n_cls], label)[0].item(),
        }

        self.update_lr()
        return loss_summary

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def after_epoch(self):
        last_epoch        = (self.epoch + 1) == self.max_epoch
        do_test           = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best     = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------

    def parse_batch_train(self, batch_x, batch_u):
        input   = batch_x["img"]
        label   = batch_x["label"]
        input_u = batch_u["img"]
        input   = input.to(self.device)
        label   = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

    # ------------------------------------------------------------------
    # Test — domain-agnostic attention inference with visual prompts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test(self, split=None):
        """Evaluate on all target domains using domain-agnostic inference.

        Text side (unchanged from base DAPL):
            Stack all n_target text-domain logit slices into [B, n_targets, n_cls],
            compute confidence = max(softmax(logits)), normalise to weights,
            then weighted-sum to [B, n_cls].

        Visual side (new):
            For each target domain i, add visual_prompts[source_count + i] to
            the raw image features before computing similarity.  The per-domain
            logit slices are then obtained as:
                feat_i  = (raw_feat + vp_i) / ||...||
                logit_i = scale * feat_i @ text_feat_i.T
            These domain logits are fed into the same confidence-weighted
            attention as above — keeping the whole pipeline domain-agnostic.

        Note: when USE_VISUAL_PROMPTS=False the visual side is a no-op and
        behaviour is identical to the original DAPL test().
        """
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT

        source_count   = self.source_count
        target_domains = self.cfg.DATASET.TARGET_DOMAINS
        n_targets      = len(target_domains)
        n_cls          = self.n_cls

        per_domain_acc = []

        for eval_target_idx, target_domain in enumerate(target_domains):
            test_loader = self._get_target_domain_test_loader(target_domain)

            correct = 0
            total   = 0

            for batch in test_loader:
                input, label = self.parse_batch_test(batch)

                with autocast():
                    # ── Raw image features (no visual prompt yet) ────────────
                    raw_feat = self.model.image_encoder(
                        input.type(self.model.dtype))   # [B, D]

                    # ── Text features for all domains ────────────────────────
                    prompts           = self.model.prompt_learner()
                    tokenized_prompts = self.model.tokenized_prompts
                    text_feat_all     = self.model.text_encoder(
                        prompts, tokenized_prompts)     # [(n_dm+1)*n_cls, D]
                    text_feat_all = (text_feat_all /
                                     text_feat_all.norm(dim=-1, keepdim=True))

                    logit_scale = self.model.logit_scale.exp()

                    # ── Per-domain logit slices ──────────────────────────────
                    # For each target domain i:
                    #   1. Add visual_prompts[source_count + i] to raw_feat
                    #   2. Normalise
                    #   3. Compute similarity against target-i text features
                    domain_logit_slices = []
                    target_indices = list(range(source_count,
                                                source_count + n_targets))

                    for i in range(n_targets):
                        dm_idx = source_count + i

                        # Visual prompt (residual shift in image space)
                        if (self.use_visual_prompts and
                                self.model.visual_prompt_learner is not None):
                            vp   = self.model.visual_prompt_learner.visual_prompts[dm_idx]
                            feat = raw_feat + vp.unsqueeze(0)   # [B, D]
                        else:
                            feat = raw_feat

                        feat = feat / feat.norm(dim=-1, keepdim=True)  # [B, D]

                        # Corresponding target-domain text features
                        t_start = dm_idx * n_cls
                        tf      = text_feat_all[t_start:t_start + n_cls]  # [n_cls, D]

                        logits_i = logit_scale * feat @ tf.t()  # [B, n_cls]
                        domain_logit_slices.append(logits_i)

                    # ── Confidence-weighted attention ────────────────────────
                    # [B, n_targets, n_cls]
                    domain_logits = torch.stack(domain_logit_slices, dim=1)

                    probs      = torch.softmax(domain_logits, dim=-1)
                    confidence = probs.max(dim=-1).values          # [B, n_targets]
                    weights    = torch.softmax(confidence, dim=1)  # [B, n_targets]

                    # Weighted sum → [B, n_cls]
                    logits = (weights.unsqueeze(-1) * domain_logits).sum(dim=1)

                pred     = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total   += label.size(0)

            acc = 100.0 * correct / total
            per_domain_acc.append(acc)
            print(f"  Domain [{target_domain}]: accuracy = {acc:.2f}%")

        mean_acc = sum(per_domain_acc) / len(per_domain_acc)
        print(f"  Mean accuracy across {n_targets} target domains: "
              f"{mean_acc:.2f}%")

        for domain, acc in zip(target_domains, per_domain_acc):
            self.write_scalar(f"test/acc_{domain}", acc, self.epoch)
        self.write_scalar("test/mean_acc", mean_acc, self.epoch)

        return mean_acc

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _get_target_domain_test_loader(self, target_domain):
        """Create a test data loader for a specific target domain."""
        from dassl.data.data_manager import DataManager

        temp_cfg = self.cfg.clone()
        temp_cfg.defrost()
        temp_cfg.DATASET.TARGET_DOMAINS = [target_domain]
        temp_cfg.freeze()

        temp_dm = DataManager(temp_cfg)
        return temp_dm.test_loader
