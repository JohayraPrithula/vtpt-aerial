import os.path as osp
import os
import datetime
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import warnings
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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class DAPL(TrainerXU):
    """Domain Adaptation via Prompt Learning(DAPL).

    Domain Adaptation via Prompt Learning
    https://arxiv.org/abs/2202.06687
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DAPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DAPL.PREC == "fp32" or cfg.TRAINER.DAPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # plus one for pseudo label per target domain
        self.n_dm = self.model.prompt_learner.n_dm + self.model.prompt_learner.n_target_domains
        self.n_cls = self.model.prompt_learner.n_cls
        self.n_target_domains = self.model.prompt_learner.n_target_domains

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
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

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DAPL.PREC == "amp" else None

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
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader_x = len(self.train_loader_x)
        
        # Handle both single target and multi-target cases
        if hasattr(self, 'train_loader_u_list'):
            # Multi-target mode
            len_train_loader_u_list = [len(loader) for loader in self.train_loader_u_list]
            
            if self.cfg.TRAIN.COUNT_ITER == "train_x":
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == "train_u":
                self.num_batches = max(len_train_loader_u_list)
            elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
                self.num_batches = min(len_train_loader_x, min(len_train_loader_u_list))
            else:
                raise ValueError

            train_loader_x_iter = iter(self.train_loader_x)
            train_loader_u_iter_list = [iter(loader) for loader in self.train_loader_u_list]
            
        else:
            # Single target mode (original behavior)
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

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            if hasattr(self, 'train_loader_u_list'):
                # Multi-target mode: collect batches from all target domains
                batch_u_list = []
                for i, train_loader_u_iter in enumerate(train_loader_u_iter_list):
                    try:
                        batch_u = next(train_loader_u_iter)
                    except StopIteration:
                        train_loader_u_iter_list[i] = iter(self.train_loader_u_list[i])
                        batch_u = next(train_loader_u_iter_list[i])
                    batch_u_list.append(batch_u)
                
                data_time.update(time.time() - end)
                loss_summary = self.forward_backward(batch_x, batch_u_list)
                
            else:
                # Single target mode: original behavior
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.train_loader_u)
                    batch_u = next(train_loader_u_iter)

                data_time.update(time.time() - end)
                loss_summary = self.forward_backward(batch_x, batch_u)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                            1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        # Handle both single target and multi-target cases
        if isinstance(batch_u, list):
            # Multi-target mode
            return self.forward_backward_multi_target(batch_x, batch_u)
        else:
            # Single target mode (original behavior)
            return self.forward_backward_single_target(batch_x, batch_u)

    def forward_backward_single_target(self, batch_x, batch_u):
        """Original single target forward_backward method."""
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.DAPL.PREC
        
        if prec == "amp":
            with autocast():
                output_x = self.model(image_x)
                output_u = self.model(image_u)

                # only clip annotation
                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1, self.n_cls) /
                    self.cfg.TRAINER.DAPL.T,
                    dim=-1)

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()

                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
                loss_u = (F.cross_entropy(
                    output_u[:, self.n_cls:2 * self.n_cls],
                    label_p,
                    reduction="none") * mask).sum() / mask.sum()
                loss = loss_x + self.cfg.TRAINER.DAPL.U * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        }

        self.update_lr()
        return loss_summary

    def forward_backward_multi_target(self, batch_x, batch_u_list):
        """Multi-target forward_backward method."""
        image_x, label = batch_x["img"].to(self.device), batch_x["label"].to(self.device)
        
        prec = self.cfg.TRAINER.DAPL.PREC
        if prec == "amp":
            with autocast():
                output_x = self.model(image_x)
                
                # Process each target domain
                loss_u_total = 0
                mask_count = 0
                
                for target_idx, batch_u in enumerate(batch_u_list):
                    image_u = batch_u["img"].to(self.device)
                    output_u = self.model(image_u)
                    
                    # Get pseudo labels from CLIP (last part of outputs)
                    source_domains_count = len(self.cfg.DATASET.SOURCE_DOMAINS)
                    total_domains = source_domains_count + len(self.cfg.DATASET.TARGET_DOMAINS)
                    clip_start_idx = self.n_cls * total_domains
                    clip_end_idx = clip_start_idx + self.n_cls
                    
                    pseudo_label = torch.softmax(
                        output_u[:, clip_start_idx:clip_end_idx] / self.cfg.TRAINER.DAPL.T,
                        dim=-1)
                    
                    max_probs, label_p = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()
                    
                    if mask.sum() > 0:
                        # Target domain specific predictions
                        target_start_idx = self.n_cls * (source_domains_count + target_idx)
                        target_end_idx = target_start_idx + self.n_cls
                        
                        loss_u_target = (F.cross_entropy(
                            output_u[:, target_start_idx:target_end_idx],
                            label_p,
                            reduction="none") * mask).sum() / mask.sum()
                        
                        loss_u_total += loss_u_target
                        mask_count += 1
                
                # Average loss across target domains
                if mask_count > 0:
                    loss_u = loss_u_total / mask_count
                else:
                    loss_u = torch.tensor(0.0).to(self.device)
                
                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
                loss = loss_x + self.cfg.TRAINER.DAPL.U * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        }

        self.update_lr()
        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """Evaluate on all target domains and return mean accuracy.

        Supports four TEST_MODE values controlled by TRAINER.DAPL.TEST_MODE:
            domain_specific : original slice-per-domain (needs domain label at test time)
            naive           : use naïve CLIP slice (Option A)
            ensemble        : average all target-domain logit slices (Option B)
            attention       : soft image-guided attention over domain heads (Option C)
        """
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT

        # Read mode — default to original behaviour if key absent
        test_mode = getattr(self.cfg.TRAINER.DAPL, "TEST_MODE", "domain_specific")

        source_count  = len(self.cfg.DATASET.SOURCE_DOMAINS)
        target_domains = self.cfg.DATASET.TARGET_DOMAINS
        n_targets      = len(target_domains)
        n_cls          = self.n_cls

        # Indices of target-domain logit slices in the flat output vector
        # output shape: [B, n_dm*n_cls + n_cls]  (last n_cls = naïve CLIP)
        target_slice_starts = [
            (source_count + i) * n_cls for i in range(n_targets)
        ]
        naive_start = n_targets * n_cls + source_count * n_cls  # == n_dm * n_cls

        print(f"\n[TEST MODE: {test_mode}]")

        per_domain_acc = []

        for target_idx, target_domain in enumerate(target_domains):
            test_loader = self._get_target_domain_test_loader(target_domain)

            correct = 0
            total   = 0

            for batch in test_loader:
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)   # [B, n_dm*n_cls + n_cls]

                # ---- choose logits based on TEST_MODE ---- #

                if test_mode == "domain_specific":
                    # Original: use the logit slice for this specific domain
                    start = target_slice_starts[target_idx]
                    logits = output[:, start : start + n_cls]          # [B, n_cls]

                elif test_mode == "naive":
                    # Option A: always use the naïve CLIP slice
                    logits = output[:, naive_start : naive_start + n_cls]  # [B, n_cls]

                elif test_mode == "ensemble":
                    # Option B: average logits across ALL target domain heads
                    # Stack → [B, n_targets, n_cls], then mean over dim=1
                    domain_logits = torch.stack([
                        output[:, s : s + n_cls]
                        for s in target_slice_starts
                    ], dim=1)                                           # [B, n_targets, n_cls]
                    logits = domain_logits.mean(dim=1)                 # [B, n_cls]

                elif test_mode == "attention":
                    # Option C: image-guided soft attention over domain heads.
                    #
                    # For each image, compute how "confident" each target-domain
                    # head is (max softmax prob across classes), then weight the
                    # logits by those confidences via softmax normalisation.
                    #
                    # Concretely:
                    #   domain_logits : [B, n_targets, n_cls]
                    #   confidence_d  : max_c softmax(domain_logits[:, d, :])  → [B, n_targets]
                    #   weights       : softmax(confidence_d, dim=1)           → [B, n_targets]
                    #   logits        : sum_d weights_d * domain_logits_d      → [B, n_cls]
                    #
                    # This lets the image itself vote for the most relevant
                    # domain head without any explicit domain label.
                    domain_logits = torch.stack([
                        output[:, s : s + n_cls]
                        for s in target_slice_starts
                    ], dim=1)                                           # [B, n_targets, n_cls]

                    # Per-domain confidence = max softmax probability
                    probs      = torch.softmax(domain_logits, dim=-1)  # [B, n_targets, n_cls]
                    confidence = probs.max(dim=-1).values              # [B, n_targets]

                    # Normalise confidences into weights
                    weights = torch.softmax(confidence, dim=1)         # [B, n_targets]

                    # Weighted sum over domain heads
                    logits = (weights.unsqueeze(-1) * domain_logits).sum(dim=1)  # [B, n_cls]

                else:
                    raise ValueError(
                        f"Unknown TEST_MODE '{test_mode}'. "
                        "Choose from: domain_specific | naive | ensemble | attention"
                    )

                # ---- accuracy accumulation ---- #
                pred     = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total   += label.size(0)

            acc = 100.0 * correct / total
            per_domain_acc.append(acc)
            print(f"  Domain [{target_domain}]: accuracy = {acc:.2f}%")

        mean_acc = sum(per_domain_acc) / len(per_domain_acc)
        print(f"  Mean accuracy across {n_targets} target domains: {mean_acc:.2f}%\n")

        # Tensorboard
        for domain, acc in zip(target_domains, per_domain_acc):
            self.write_scalar(f"test/{test_mode}/acc_{domain}", acc, self.epoch)
        self.write_scalar(f"test/{test_mode}/mean_acc", mean_acc, self.epoch)

        return mean_acc

    def _get_target_domain_test_loader(self, target_domain):
        """Create a test data loader for a specific target domain."""
        from dassl.data.data_manager import DataManager

        temp_cfg = self.cfg.clone()
        temp_cfg.defrost()
        temp_cfg.DATASET.TARGET_DOMAINS = [target_domain]
        temp_cfg.freeze()

        temp_dm = DataManager(temp_cfg)
        return temp_dm.test_loader