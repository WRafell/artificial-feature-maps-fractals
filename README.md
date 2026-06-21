# Artificial Class Activation Maps using Fractals

Code for *"Fractal-Based Artificial Class Activation Maps for Data Augmentation in Whole-Slide Image Analysis"*
(Quinones Robles, Noree, Ko, Yi — KAIST + Seegene Medical Foundation).

## TL;DR

Whole-slide image (WSI) classification needs many labelled slides. Generating synthetic *images* (GANs, Diffusion) is expensive and brittle. We instead synthesize the **Class Activation Maps (CAMs)** that the slide-level model actually consumes:

- **Tissue shape** ← Julia-set fractals.
- **Patch confidence scores** ← sampled from the patch classifier's known distribution.

No generative network. Pure heuristics. Plug the artificial CAMs into the training pool of a ResNet50 slide classifier.

With only **25 real maps + 150 fractal maps**, stomach accuracy goes from **62.86 % → 92.35 %** and AUC from **82.34 % → 96.92 %**. Same fractal pool transfers from stomach to colorectal without regeneration, and the method holds across patch backbones (ResNet-50, ViT-Tiny, CTransPath) and on the public Camelyon16 benchmark.

## Pipeline

```
WSI → patches → patch classifier → real CAMs ─┐
                                              ├─→ slide classifier (ResNet50 / ViT-Tiny)
                  fractals + sampled scores ──┘
```

1. **Patch classifier** (ResNet50, 3 classes N/D/M) — `train_patch_classifier.py`
2. **Real CAMs**: run patches through patch classifier, place softmax scores in a `(num_classes, H, W)` tensor at each patch's grid coord — `generate_feature_maps.py` → `src/utils_feature_maps.py::build_feature_map`
3. **Artificial CAMs**: Julia fractals + sampled confidence scores — `generate_artificial_feature_maps.py` → `src/utils_artificial_maps.py`
4. **Slide classifier** trained on real CAMs alone (baseline) or real + artificial mix:
   - `train_baseline_slide_classifier.py`
   - `train_artificial_maps.py` (real + fractal)
   - `train_patch_augmented_maps.py` (real + patch-level augmentation, ablation)
5. **MIL comparison** (ABMIL, GABMIL, DSMIL, TransMIL, DTFD-MIL, max/mean pool) — `mil_training.py` + `src/mil/` after extracting latents with `extract_latent.py`.

## Setup

```bash
conda create -n histo python=3.11 -y
conda activate histo
pip install -r requirements.txt
```

CUDA-enabled GPU recommended. The pinned `torch`/`torchvision` assume a recent CUDA runtime; adjust to your platform if needed.

**CTransPath backbone (optional).** To use the CTransPath patch encoder, download the pretrained `ctranspath.pth` weights from the official [TransPath repository](https://github.com/Xiyue-Wang/TransPath) (Wang et al.) and place them at `models/ctranspath.pth`. CTransPath is loaded frozen with a linear-probe head (`src/encoders/ctranspath.py`); `timm==0.5.4` is pinned because the model uses a custom ConvStem patch embedding, which we swap in manually after `create_model` (newer `timm` ignores the `embed_layer` kwarg). The weights are not redistributed here — see the TransPath repository for the download link and license.

## Repo layout

```
.
├── csv/                                splits per organ × slides_per_class
├── src/
│   ├── utils_artificial_maps.py        Julia fractal + artificial CAM generator
│   ├── utils_feature_maps.py           real CAM builder, datasets, transforms
│   ├── utils_patch_classifier.py       patch model train/test loops
│   ├── utils_slide_classifier.py       slide model train/test loops
│   ├── utils_trainer.py
│   ├── encoders/ctranspath.py          CTransPath (Swin-Tiny + ConvStem) encoder + classifier
│   ├── analysis/cam_shape_statistics.py  quantitative real-vs-artificial CAM shape descriptors
│   └── mil/                            ABMIL / GABMIL / DSMIL / TransMIL / DTFD-MIL / CLAM
├── train_patch_classifier.py           patch classifier (ResNet-50 / ViT-Tiny / CTransPath)
├── generate_feature_maps.py            real CAMs
├── generate_artificial_feature_maps.py        fractal CAMs
├── generate_artificial_feature_maps_sweep.py  score-design variants (sensitivity ablation)
├── generate_patch_augmented_maps.py    augmented CAMs (ablation)
├── extract_latent.py                   patch latents for MIL
├── train_baseline_slide_classifier.py
├── train_artificial_maps.py            real + fractal (patch/slide backbone selectable)
├── train_artificial_sweep.py           score-design sensitivity experiments
├── train_patch_augmented_maps.py
├── mil_training.py
└── camelyon16_exp/                     standalone cross-organ validation on Camelyon16
    ├── utils_camelyon16.py
    ├── train_camelyon16_patch_classifier.py
    ├── generate_camelyon16_feature_maps.py
    ├── generate_artificial_camelyon16_feature_maps.py
    ├── extract_camelyon16_latent.py
    └── train_artificial_camelyon16.py
```

## Run order (Seegene stomach / colorectal)

```bash
# 1. patch classifier per (organ, slides_per_class)
python train_patch_classifier.py

# 2. real CAMs (uses trained patch classifier)
python generate_feature_maps.py

# 3. fractal CAMs (independent of patch classifier)
python generate_artificial_feature_maps.py

# 4a. baseline slide model (real only)
python train_baseline_slide_classifier.py

# 4b. real + fractal
python train_artificial_maps.py

# 4c. real + patch-augmented (ablation)
python generate_patch_augmented_maps.py
python train_patch_augmented_maps.py

# 5. MIL baselines
python extract_latent.py
python mil_training.py
```

Logs land under `logs/`, models under `models/`, feature maps under `local/feature_maps/`.

### Alternative patch / slide backbones

`train_patch_classifier.py` and `train_artificial_maps.py` expose `BACKBONE_NAME` (patch) and `SLIDE_BACKBONE_NAME` (slide) constants. ViT-Tiny pairs a ViT-Tiny slide model (learning rate `1e-4`); ResNet-50 and CTransPath pair a ResNet-50 slide model (learning rate `1e-3`). Already-trained `(patch, slide, real, artificial)` combinations are skipped on resume.

### Confidence-score sensitivity ablation

```bash
python generate_artificial_feature_maps_sweep.py   # generates score-design variants
python train_artificial_sweep.py                   # sweeps positive floor / noise / score mode
```

### Shape-descriptor analysis (real vs. artificial CAMs)

```bash
python -m src.analysis.cam_shape_statistics        # writes CSV + summary to local/analysis/
```

### Camelyon16 cross-organ validation

```bash
python camelyon16_exp/train_camelyon16_patch_classifier.py
python camelyon16_exp/generate_camelyon16_feature_maps.py
python camelyon16_exp/generate_artificial_camelyon16_feature_maps.py
python camelyon16_exp/extract_camelyon16_latent.py
python camelyon16_exp/train_artificial_camelyon16.py
```

50 normal + 50 tumor training slides; the official Camelyon16 validation (55) and test (129) splits are kept unchanged.

## Key constants (paper defaults)

| Setting             | Value                              |
|---------------------|------------------------------------|
| Patch backbones     | ResNet-50 (ImageNet), ViT-Tiny, CTransPath |
| Slide backbones     | ResNet-50, ViT-Tiny                |
| Classes             | N (negative), D (dysplasia), M (malignant) |
| Patch size          | 256 × 256 @ 200× magnification     |
| CAM resolution      | 256 × 128                          |
| Fractal slices/CAM  | 2–4 (p = 0.15 / 0.7 / 0.15)        |
| Fractal scale       | U(0.4, 0.95), escape radius 2      |
| Positive score      | U(0.8, 1), Gaussian noise std 0.1  |
| Optimizer           | Adam, lr 1e-3 (1e-4 for ViT-Tiny), wd 5e-4 |
| Patch model         | batch 128, 50 epochs               |
| Slide model         | batch 128, 100 epochs, ES patience 3 |
| Seeds               | 5 runs, base seed 2024             |

## Data

Stomach + colorectal WSIs, 285 per class (N/D/M), provided by Seegene Medical Foundation (KR). **Not redistributable** — see the paper's data availability statement. CSV splits in `csv/` reference paths to local patch directories. Each WSI corresponds to a distinct patient, so the train/validation/test split is patient-level.

[Camelyon16](https://camelyon16.grand-challenge.org/) is a public breast-lymph-node benchmark; download it from the challenge site.

## Citation

```bibtex
@article{quinonesrobles_artificialcams,
  title  = {Fractal-Based Artificial Class Activation Maps for Data Augmentation in Whole-Slide Image Analysis},
  author = {Quinones Robles, Willmer Rafell and Noree, Sakonporn and
            Ko, Young Sin and Yi, Mun Yong},
  year   = {2025}
}
```

Funding: Seegene Medical Foundation, Grant G01180115.
