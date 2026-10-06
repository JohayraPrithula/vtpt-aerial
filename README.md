# VTPT-Aerial

Code for the paper *Visual-Textual Prompt Tuning for Unsupervised Multi-Target Domain Adaptation in Aerial Scene Classification* (Johayra Prithula and Andreas Savakis, Rochester Institute of Technology), submitted to the Journal of Applied Remote Sensing.
<!-- CHECK: update the title here if the paper title changes -->

This repository extends VTPT (CVPR 2026) from natural images to aerial imagery. A frozen CLIP model is adapted by learning prompts in both encoders: shared and domain-specific text tokens on the text side, and a per-domain residual vector added to the image embedding on the visual side. A single model is adapted from one labeled source dataset to two unlabeled target datasets at once. At test time, predictions from all domain heads are combined by confidence, so the dataset an image came from does not need to be known. Only the prompts are trained; the CLIP weights are never updated.

## Setup

The code was tested with Python 3.9, PyTorch 2.8, and CUDA 12.8.

```bash
conda create -n vtpt python=3.9 -y
conda activate vtpt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The trainer is built on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), which has to be installed first:

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

Then install the remaining requirements from the root of this repository:

```bash
pip install -r requirements.txt
```

CLIP weights (RN50, RN101, ViT-B/16) are downloaded on first use.

## Data

We use three public benchmarks and treat each one as a separate domain. The images are not redistributed here. Please download them from the original providers and cite the corresponding papers.

| Dataset | Source |
|---|---|
| AID | https://captain-whu.github.io/AID/ |
| CLRS | https://github.com/lehaifeng/CLRS |
| NWPU-RESISC45 | https://doi.org/10.1109/JPROC.2017.2675998 |

Place the datasets under one root directory, keeping the original folder names:

```
DATA_ROOT/
├── AID_dataset/
│   └── data/
│       ├── Airport/
│       └── ...
├── CLRS_dataset/
│   ├── airport/
│   └── ...
└── NWPU_dataset/
    ├── airport/
    └── ...
```

Only `AID_dataset/data` is read for AID. A `.cache` folder inside `AID_dataset` (created when AID is downloaded through Hugging Face) is ignored.

### Label space

The three datasets use different class names for the same categories (for example, NWPU-RESISC45 uses *harbor* where the others use *port*, and AID uses *StorageTanks*). `datasets/aerial_class_map.py` maps every dataset folder to a shared vocabulary, and the experiments use the 12 classes common to the three datasets:

airport, beach, bridge, forest, industrial, mountain, parking, port, railway station, river, stadium, storage tank

The list is fixed in `configs/datasets/aerial_acn12.yaml` (`SHARED_CLASSES`) so that the source loader and every target loader use the same classes and the same label indices.

## Usage

Set `DATA` and `PYTHON` at the top of `scripts/main.sh` and `scripts/eval.sh`. Both scripts are run from inside `scripts/`:

```bash
cd scripts
bash main.sh DATASET_CFG TRAINER_CFG SOURCE "TARGETS" T TAU U NAME
```

`T` is the softmax temperature, `TAU` is the pseudo-label confidence threshold, and `U` is the weight of the target loss. Outputs are written to `output/<DATASET_CFG>/DAPL/<TRAINER_CFG>/<SOURCE>_to_<TARGETS>/<T>_<TAU>_<U>_<NAME>/seed_1/`. The trainer resumes from an existing checkpoint in that folder, so use a new `NAME` to start from scratch.

Multi-target adaptation, one command per source dataset:

```bash
bash main.sh aerial_acn12 ep25-32 AID  "CLRS NWPU" 1.0 0.8 0.5 vtpt
bash main.sh aerial_acn12 ep25-32 CLRS "AID NWPU"  1.0 0.8 0.5 vtpt
bash main.sh aerial_acn12 ep25-32 NWPU "AID CLRS"  1.0 0.8 0.5 vtpt
```

Trainer configs:

| Config | Description |
|---|---|
| `ep25-32` | VTPT, ResNet-50 |
| `ep25-32-rn101` | VTPT, ResNet-101 |
| `ep25-32-vitb16` | VTPT, ViT-B/16 |
| `ep25-32-csc` | Class-specific context instead of shared context |
| `ep25-32-vpoff` | Text prompts only (visual prompts disabled) |
| `ep25-32-novp`, `-novp-rn101`, `-novp-vitb16` | DAPL baseline |

DAPL is a single-target method, so for the baseline the two target datasets are merged into one target domain using the `aerial_lump_acn12` dataset config:

```bash
bash main.sh aerial_lump_acn12 ep25-32-novp AID "REST" 1.0 0.7 1.0 dapl
```

To evaluate a trained model, call `eval.sh` with the same arguments used for training:

```bash
bash eval.sh aerial_acn12 ep25-32 AID "CLRS NWPU" 1.0 0.8 0.5 vtpt
```

Zero-shot CLIP, from the repository root:

```bash
python zero_shot_eval.py --root /path/to/DATA_ROOT \
    --dataset-config-file configs/datasets/aerial_acn12.yaml \
    --config-file configs/trainers/DAPL/ep25-32.yaml \
    --source-domains AID --target-domains CLRS NWPU
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{prithula2026visual,
  title     = {Visual-Textual Prompt Tuning for Unsupervised Multi-Domain Adaptation},
  author    = {Prithula, Johayra and Savakis, Andreas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {9040--9049},
  year      = {2026}
}
```

## Acknowledgments

This code builds on [DAPL](https://github.com/LeapLabTHU/DAPrompt), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), and [CLIP](https://github.com/openai/CLIP). We thank the authors of AID, CLRS, and NWPU-RESISC45 for making their datasets available.

## License

See [LICENSE](LICENSE).
