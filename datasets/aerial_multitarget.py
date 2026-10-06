import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from datasets.aerial_class_map import (
    AERIAL_DATASET_DIRS,
    FOLDER_TO_CANONICAL,
    compute_shared_classes,
    get_canonical_classes,
)

# Image extensions across the four datasets (UCM is .tif, others mostly .jpg).
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@DATASET_REGISTRY.register()
class AerialMultiTarget(DatasetBase):
    """Cross-dataset aerial scene classification — closed-set multi-target DA.

    Each "domain" is a separate aerial dataset directory (AID, CLRS, NWPU,
    UCM) rather than a sub-folder within one dataset.  Only the classes
    shared across the active source + target domains are used (closed-set
    intersection).  Folder names are normalised to a canonical vocabulary
    (see aerial_class_map.py) and re-indexed to a contiguous 0..C-1 label
    space that is identical across every domain, so a label means the same
    class no matter which dataset it came from.

    cfg.DATASET.ROOT should point at the parent folder that contains
    AID_dataset/, CLRS_dataset/, NWPU_dataset/, UCM_dataset/
    (e.g. D:/VIP/Dataset/MultiDomainAerial).
    """

    dataset_dir = "MultiDomainAerial"  # informational only; see per-domain dirs
    domains = ["AID", "CLRS", "NWPU", "UCM"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.root = root

        source_domains = list(cfg.DATASET.SOURCE_DOMAINS)
        target_domains = list(cfg.DATASET.TARGET_DOMAINS)

        self.check_input_domains(source_domains, target_domains)

        # ── Closed-set label space ────────────────────────────────────────
        # For TRUE multi-target, every per-target loader must use the SAME
        # class list, otherwise each (source, single-target) pair would
        # recompute a larger *pairwise* intersection and pull off-task
        # images into train_u.  cfg.DATASET.SHARED_CLASSES, when non-empty,
        # forces a fixed closed set across every loader.  When empty, we
        # fall back to the dynamic intersection over the involved domains.
        involved = source_domains + target_domains
        forced = list(getattr(cfg.DATASET, "SHARED_CLASSES", []) or [])

        if forced:
            shared_classes = self._validate_forced_classes(forced, involved)
            mode = "forced SHARED_CLASSES"
        else:
            shared_classes = compute_shared_classes(involved)
            mode = "auto-intersection"

        if len(shared_classes) == 0:
            raise RuntimeError(
                f"No shared classes across domains {involved}. "
                "Check datasets/aerial_class_map.py or cfg.DATASET.SHARED_CLASSES."
            )
        self.shared_classes = shared_classes
        self._cname2lab = {c: i for i, c in enumerate(shared_classes)}

        print(f"[AerialMultiTarget] Source: {source_domains} | "
              f"Targets: {target_domains}")
        print(f"[AerialMultiTarget] Closed-set classes [{mode}] "
              f"({len(shared_classes)}): {shared_classes}")

        train_x = self._read_data(source_domains)
        train_u = self._read_data(target_domains)
        test = self._read_data(target_domains)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _validate_forced_classes(self, forced, involved):
        """Validate an explicit SHARED_CLASSES list and return it sorted.

        Sorting yields a deterministic, contiguous 0..C-1 label ordering that
        is identical no matter the order the classes are listed in the config.
        Every involved domain must contain every forced class, otherwise that
        domain would silently contribute zero images for the missing class and
        the closed-set assumption would be violated.
        """
        forced_sorted = sorted(set(forced))
        for dname in involved:
            avail = get_canonical_classes(dname)
            missing = [c for c in forced_sorted if c not in avail]
            if missing:
                raise ValueError(
                    f"cfg.DATASET.SHARED_CLASSES contains classes not present "
                    f"in domain '{dname}': {missing}. Available canonical "
                    f"classes for {dname}: {sorted(avail)}"
                )
        return forced_sorted

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            if dname not in AERIAL_DATASET_DIRS:
                raise ValueError(f"Unknown aerial domain: {dname}")

            subdir = AERIAL_DATASET_DIRS[dname]
            domain_dir = osp.join(self.root, subdir)
            folder_map = FOLDER_TO_CANONICAL[dname]

            if not osp.isdir(domain_dir):
                raise FileNotFoundError(
                    f"Domain directory not found for {dname}: {domain_dir}"
                )

            folder_names = listdir_nohidden(domain_dir)
            folder_names.sort()

            for folder_name in folder_names:
                # Map on-disk folder -> canonical class name
                canonical = folder_map.get(folder_name, None)
                if canonical is None:
                    # Folder is not in our canonical map (unknown class) -> skip
                    continue
                if canonical not in self._cname2lab:
                    # Canonical class is not part of the shared closed set -> skip
                    continue

                label = self._cname2lab[canonical]
                class_path = osp.join(domain_dir, folder_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    if not imname.lower().endswith(IMG_EXTENSIONS):
                        continue
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=canonical,
                    )
                    items.append(item)

        return items
