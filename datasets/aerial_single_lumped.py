import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from datasets.aerial_class_map import (
    AERIAL_DATASET_DIRS,
    FOLDER_TO_CANONICAL,
    compute_shared_classes,
    get_canonical_classes,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@DATASET_REGISTRY.register()
class AerialSingleLumped(DatasetBase):
    """DAPrompt-style single-source -> single-target baseline (aerial).

    This is the closed-set cross-dataset analogue of the original DAPrompt
    setting: exactly TWO domains (source, target). All aerial datasets that
    are NOT the source are POOLED into one combined target domain, so the
    model learns a single target prompt (t^u) over the union, exactly as
    DAPrompt does with 1 source + 1 target.

    Differences from AerialMultiTarget:
      * The target is a single lumped domain (domain index 1), not K
        separate targets. Every pooled target image gets domain=1.
      * TARGET_DOMAINS in the config is a placeholder (e.g. ["REST"]); the
        actual pooled datasets are derived automatically as
        (all four datasets) - (source). This keeps len(TARGET_DOMAINS)==1
        so train.py routes to the single-target path.

    Use the same forced closed set (cfg.DATASET.SHARED_CLASSES = the 6
    classes) so the numbers sit directly next to the multi-target table.
    Run with TRAINER.DAPL.USE_VISUAL_PROMPTS = False for a pure DAPrompt
    baseline (no visual prompts).
    """

    dataset_dir = "MultiDomainAerial"  # informational
    domains = ["AID", "CLRS", "NWPU", "UCM"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.root = root

        source_domains = list(cfg.DATASET.SOURCE_DOMAINS)

        # Which datasets participate in this family. Defaults to all four.
        # Set cfg.DATASET.LUMP_DOMAINS to restrict the family, e.g.
        # ["AID","CLRS","NWPU"] to exclude UCM.
        family = list(getattr(cfg.DATASET, "LUMP_DOMAINS", []) or self.domains)

        # Lumped target = every family dataset except the source(s).
        lumped_targets = [d for d in family if d not in source_domains]

        # Validate the real domains (not the placeholder in TARGET_DOMAINS).
        for d in source_domains + lumped_targets:
            if d not in self.domains:
                raise ValueError(f"Unknown aerial domain: {d}")

        # ── Closed-set label space (same forced 6 classes) ────────────────
        involved = source_domains + lumped_targets
        forced = list(getattr(cfg.DATASET, "SHARED_CLASSES", []) or [])
        if forced:
            shared_classes = self._validate_forced_classes(forced, involved)
            mode = "forced SHARED_CLASSES"
        else:
            shared_classes = compute_shared_classes(involved)
            mode = "auto-intersection"

        if len(shared_classes) == 0:
            raise RuntimeError(
                f"No shared classes across {involved}. "
                "Check aerial_class_map.py or cfg.DATASET.SHARED_CLASSES."
            )
        self.shared_classes = shared_classes
        self._cname2lab = {c: i for i, c in enumerate(shared_classes)}

        print(f"[AerialSingleLumped] Source: {source_domains} | "
              f"Lumped target: {lumped_targets}")
        print(f"[AerialSingleLumped] Closed-set classes [{mode}] "
              f"({len(shared_classes)}): {shared_classes}")

        # Source -> domain index 0; ALL pooled targets -> domain index 1.
        train_x = self._read_data(source_domains, force_domain=0)
        train_u = self._read_data(lumped_targets, force_domain=1)
        test = self._read_data(lumped_targets, force_domain=1)

        print(f"[AerialSingleLumped] # source imgs {len(train_x)} | "
              f"# pooled target imgs {len(train_u)}")

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def check_input_domains(self, source_domains, target_domains):
        # TARGET_DOMAINS is a placeholder here (e.g. "REST"); the real domain
        # validation is done in __init__. Override to a no-op so the base
        # class does not reject the placeholder name.
        return

    def _validate_forced_classes(self, forced, involved):
        forced_sorted = sorted(set(forced))
        for dname in involved:
            avail = get_canonical_classes(dname)
            missing = [c for c in forced_sorted if c not in avail]
            if missing:
                raise ValueError(
                    f"cfg.DATASET.SHARED_CLASSES contains classes not present "
                    f"in domain '{dname}': {missing}. Available for {dname}: "
                    f"{sorted(avail)}"
                )
        return forced_sorted

    def _read_data(self, input_domains, force_domain):
        """Read images from `input_domains`, assigning EVERY item the same
        `force_domain` index (0 for source, 1 for the lumped target)."""
        items = []

        for dname in input_domains:
            subdir = AERIAL_DATASET_DIRS[dname]
            domain_dir = osp.join(self.root, subdir)
            folder_map = FOLDER_TO_CANONICAL[dname]

            if not osp.isdir(domain_dir):
                raise FileNotFoundError(
                    f"Domain directory not found for {dname}: {domain_dir}"
                )

            for folder_name in sorted(listdir_nohidden(domain_dir)):
                canonical = folder_map.get(folder_name, None)
                if canonical is None or canonical not in self._cname2lab:
                    continue

                label = self._cname2lab[canonical]
                class_path = osp.join(domain_dir, folder_name)
                for imname in listdir_nohidden(class_path):
                    if not imname.lower().endswith(IMG_EXTENSIONS):
                        continue
                    items.append(
                        Datum(
                            impath=osp.join(class_path, imname),
                            label=label,
                            domain=force_domain,
                            classname=canonical,
                        )
                    )

        return items
