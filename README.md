# Artificial Class Activation Maps using Fractals

Code for *"Artificial Class Activation Maps Using Fractals: A New Data Augmentation Strategy for Deep Learning-based Whole-Slide Image Analysis"*
(Quinones Robles, Noree, Ko, Yi — KAIST + Seegene Medical Foundation).

## TL;DR

Whole-slide image (WSI) classification needs many labelled slides. Generating synthetic *images* (GANs, Diffusion) is expensive and brittle. We instead synthesize the **Class Activation Maps (CAMs)** that the slide-level model actually consumes:

- **Tissue shape** ← Julia-set fractals.
- **Patch confidence scores** ← sampled from the patch classifier's known distribution.

No generative network. Pure heuristics. Plug the artificial CAMs into the training pool of a ResNet50 slide classifier.

With only **25 real maps + 150 fractal maps**, stomach accuracy goes from **62.86 % → 92.35 %** and AUC from **82.34 % → 96.92 %**. Same fractal pool transfers from stomach to colorectal without regeneration.

## Pipeline

```
WSI → patches → patch classifier → real CAMs ─┐
                                              ├─→ slide classifier (ResNet50)
                  fractals + sampled scores ──┘
```

1. **Patch classifier** (ResNet50, 3 classes N/D/M) — `train_patch_classifier.py`
2. **Real CAMs**: run patches through patch classifier, place softmax scores in a `(num_classes, H, W)` tensor at each patch's grid coord — `generate_feature_maps.py` → uses `src/utils_feature_maps.py::build_feature_map`
3. **Artificial CAMs**: Julia fractals + sampled confidence scores — `generate_artificial_feature_maps.py` → uses `src/utils_artificial_maps.py`
4. **Slide classifier** (ResNet50) trained on real CAMs alone (baseline) or real + artificial mix:
   - `train_baseline_slide_classifier.py`
   - `train_artificial_maps.py` (real + fractal)
   - `train_patch_augmented_maps.py` (real + patch-level augmentation, ablation)
5. **MIL comparison** (ABMIL, GABMIL, DSMIL, TransMIL, DTFD-MIL, max/mean pool) — `mil_training.py` + `src/mil/` after extracting latents with `extract_latent.py`.

## Repo layout

```
.
├── csv/                              splits per organ × slides_per_class
├── src/
│   ├── utils_artificial_maps.py      Julia fractal + artificial CAM generator
│   ├── utils_feature_maps.py         real CAM builder, datasets, transforms
│   ├── utils_patch_classifier.py     patch ResNet50 train/test loops
│   ├── utils_slide_classifier.py     slide ResNet50 train/test loops
│   ├── utils_trainer.py
│   ├── mil/                          ABMIL / GABMIL / DSMIL / TransMIL / DTFD-MIL / CLAM
├── train_patch_classifier.py
├── generate_feature_maps.py          real CAMs
├── generate_artificial_feature_maps.py  fractal CAMs
├── generate_patch_augmented_maps.py  augmented CAMs (ablation)
├── extract_latent.py                 patch latents for MIL
├── train_baseline_slide_classifier.py
├── train_artificial_maps.py
├── train_patch_augmented_maps.py
├── mil_training.py
├── data_distribution.ipynb
├── playground.ipynb
```

## Run order

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

Logs land under `logs/`. Models under `models/`. Feature maps under `feature_maps/`.

## Key constants (paper defaults)

| Setting             | Value                              |
|---------------------|------------------------------------|
| Backbone            | ResNet50 (ImageNet init)           |
| Classes             | N (negative), D (dysplasia), M (malignant) |
| Patch size          | 256 × 256 @ 200× magnification     |
| CAM resolution      | 256 × 128                          |
| Fractal slices/CAM  | 2–4 (p = 0.15 / 0.7 / 0.15)        |
| Fractal scale       | U(0.4, 0.95), escape radius 2      |
| Optimizer           | Adam, lr 1e-3, wd 5e-4             |
| Patch model         | batch 128, 50 epochs               |
| Slide model         | batch 128, 100 epochs, ES patience 3 |
| Seeds               | 5 runs, base seed 2024             |

## Data

Stomach + colorectal WSIs, 900 each, 300 per class N/D/M. Provided by Seegene Medical Foundation (KR). **Not redistributable** — see paper's data availability statement. CSV splits in `csv/` reference paths to local patch directories.

## Citation

```
Quinones Robles, W. R., Noree, S., Ko, Y. S., & Yi, M. Y.
Artificial Class Activation Maps Using Fractals: A New Data Augmentation Strategy
for Deep Learning-based Whole-Slide Image Analysis.
```

Funding: Seegene Medical Foundation, Grant G01180115.
