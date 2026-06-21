"""Quantitative comparison of real vs. artificial (fractal) vs. simple-shape CAMs.

Addresses Reviewer 4, Comment 1 (June-16 round): motivate the use of fractals
beyond visual similarity and downstream accuracy by quantifying how closely
artificial CAMs reproduce the *shape statistics* of real CAMs.

For every CAM we reduce the (K, H, W) score tensor to a binary tissue support
(any class score above a threshold) and compute five shape descriptors:

  1. tissue_area_fraction   - fraction of non-zero (tissue) pixels.
  2. n_connected_components  - number of separate tissue blobs.
  3. fractal_dimension       - box-counting (Minkowski-Bouligand) dimension of
                               the tissue boundary; captures contour complexity.
  4. compactness             - perimeter^2 / (4*pi*area); 1.0 for a disk, larger
                               for irregular/branched shapes.
  5. nn_distance             - mean nearest-neighbor distance between connected
                               components (spatial clustering of tissue pieces).

Real and fractal CAMs are read from disk. Simple geometric shapes
(ellipse / triangle / quadrilateral / polygon) are generated inline on a
matching canvas so the comparison is fully self-contained and does not depend
on the shape-ablation CAMs being present on disk.

Outputs:
  - prints a summary table (mean +/- std per descriptor x group)
  - writes a CSV of per-CAM descriptors
  - writes box/violin plots comparing the groups

Pure NumPy / SciPy / scikit-image / matplotlib. No GPU, no model loading.

Run from the project root:
    python -m src.analysis.cam_shape_statistics
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import draw, measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_SEED = 2024

# CAM canvas (matches the main pipeline: 256 high x 128 wide, 3 classes).
LSIZE, CSIZE = 256, 128
NUM_FEATURES = 3

# Score threshold defining the tissue support (any class score above this).
SUPPORT_THRESHOLD = 0.05

# How many CAMs to sample per group (keep runtime bounded; None = all).
N_PER_GROUP = 300

# Real CAMs: aggregate the abnormal classes (D, M) since negative CAMs are
# essentially empty matrices of zeros and carry no shape.
REAL_CAM_DIRS = [
    f"local/feature_maps/baseline/resnet50/stomach/140/train/D",
    f"local/feature_maps/baseline/resnet50/stomach/140/train/M",
    f"local/feature_maps/baseline/resnet50/colon/140/train/D",
    f"local/feature_maps/baseline/resnet50/colon/140/train/M",
]

# Fractal artificial CAMs (abnormal classes only).
FRACTAL_CAM_DIRS = [
    "local/feature_maps/artificial/fractals/D",
    "local/feature_maps/artificial/fractals/M",
]

OUT_DIR = Path("local/analysis/cam_shape_statistics")
OUT_CSV = OUT_DIR / "cam_shape_descriptors.csv"
OUT_PLOT = OUT_DIR / "cam_shape_descriptors.png"

DESCRIPTORS = [
    "tissue_area_fraction",
    "n_connected_components",
    "fractal_dimension",
    "compactness",
    "nn_distance",
]


# --------------------------------------------------------------------------- #
# CAM loading / support extraction
# --------------------------------------------------------------------------- #
def load_cam_support(path: Path) -> np.ndarray:
    """Load a pickled CAM tensor (1, K, H, W) -> binary support (H, W)."""
    with path.open("rb") as f:
        arr = np.array(pickle.load(f))
    # (1, K, H, W) -> (K, H, W)
    if arr.ndim == 4:
        arr = arr[0]
    score_max = arr.max(axis=0)  # (H, W) max over classes
    return score_max > SUPPORT_THRESHOLD


def iter_supports_from_dirs(dirs: list[str], rng: np.random.RandomState) -> list[np.ndarray]:
    files: list[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"[warn] missing dir: {d}")
            continue
        files.extend(sorted(f for f in p.iterdir() if f.suffix == ".txt"))
    if not files:
        return []
    if N_PER_GROUP is not None and len(files) > N_PER_GROUP:
        idx = rng.choice(len(files), size=N_PER_GROUP, replace=False)
        files = [files[i] for i in idx]
    supports = []
    for f in files:
        try:
            s = load_cam_support(f)
            if s.sum() > 0:
                supports.append(s)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed {f}: {e}")
    return supports


# --------------------------------------------------------------------------- #
# Inline generation of simple geometric-shape supports (self-contained)
# --------------------------------------------------------------------------- #
def _blank() -> np.ndarray:
    return np.zeros((LSIZE, CSIZE), dtype=bool)


def gen_ellipse_support(rng: np.random.RandomState) -> np.ndarray:
    img = _blank()
    cy, cx = rng.randint(LSIZE // 3, 2 * LSIZE // 3), rng.randint(CSIZE // 3, 2 * CSIZE // 3)
    ry = rng.randint(LSIZE // 8, LSIZE // 3)
    rx = rng.randint(CSIZE // 8, CSIZE // 3)
    rr, cc = draw.ellipse(cy, cx, ry, rx, shape=img.shape)
    img[rr, cc] = True
    return img


def _polygon_support(rng: np.random.RandomState, n_vertices: int) -> np.ndarray:
    img = _blank()
    cy, cx = LSIZE // 2, CSIZE // 2
    radius = rng.randint(min(LSIZE, CSIZE) // 4, min(LSIZE, CSIZE) // 2)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_vertices))
    ys = cy + radius * np.sin(angles) * (LSIZE / min(LSIZE, CSIZE))
    xs = cx + radius * np.cos(angles)
    rr, cc = draw.polygon(ys, xs, shape=img.shape)
    img[rr, cc] = True
    return img


def gen_triangle_support(rng: np.random.RandomState) -> np.ndarray:
    return _polygon_support(rng, 3)


def gen_quadrilateral_support(rng: np.random.RandomState) -> np.ndarray:
    return _polygon_support(rng, 4)


def gen_polygon_support(rng: np.random.RandomState) -> np.ndarray:
    return _polygon_support(rng, rng.randint(5, 9))


def gen_simple_shapes(generator, n: int, rng: np.random.RandomState) -> list[np.ndarray]:
    return [generator(rng) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Shape descriptors
# --------------------------------------------------------------------------- #
def fractal_dimension(support: np.ndarray) -> float:
    """Minkowski-Bouligand (box-counting) dimension of the support boundary."""
    # Use the boundary so the descriptor measures contour complexity, not fill.
    boundary = support ^ ndimage.binary_erosion(support)
    pts = boundary
    if pts.sum() < 2:
        return 0.0

    # Pad to the next power of two so boxes tile evenly.
    h, w = pts.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=bool)
    padded[:h, :w] = pts

    counts, scales = [], []
    box = size
    while box >= 2:
        n_boxes = 0
        for i in range(0, size, box):
            for j in range(0, size, box):
                if padded[i:i + box, j:j + box].any():
                    n_boxes += 1
        counts.append(n_boxes)
        scales.append(box)
        box //= 2

    counts = np.array(counts, dtype=float)
    scales = np.array(scales, dtype=float)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    # slope of log(N) vs log(1/scale)
    coeffs = np.polyfit(np.log(1.0 / scales[valid]), np.log(counts[valid]), 1)
    return float(coeffs[0])


def compactness(support: np.ndarray) -> float:
    """perimeter^2 / (4*pi*area). 1.0 for a perfect disk; larger = more irregular."""
    area = float(support.sum())
    if area == 0:
        return np.nan
    perim = float(measure.perimeter(support, neighborhood=8))
    if perim == 0:
        return np.nan
    return (perim ** 2) / (4.0 * np.pi * area)


def nn_distance(labeled: np.ndarray, n_components: int) -> float:
    """Mean nearest-neighbor distance between connected-component centroids."""
    if n_components < 2:
        return 0.0
    centroids = ndimage.center_of_mass(
        np.ones_like(labeled), labeled, index=range(1, n_components + 1)
    )
    cents = np.array(centroids)
    dists = []
    for i in range(len(cents)):
        d = np.linalg.norm(cents - cents[i], axis=1)
        d[i] = np.inf
        dists.append(d.min())
    return float(np.mean(dists))


def describe_support(support: np.ndarray) -> dict:
    area_frac = float(support.mean())
    labeled, n_comp = ndimage.label(support)
    return {
        "tissue_area_fraction": area_frac,
        "n_connected_components": int(n_comp),
        "fractal_dimension": fractal_dimension(support),
        "compactness": compactness(support),
        "nn_distance": nn_distance(labeled, n_comp),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def collect_group(name: str, supports: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for s in supports:
        d = describe_support(s)
        d["group"] = name
        rows.append(d)
    print(f"  {name:14s}: {len(rows)} CAMs")
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.RandomState(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CAM supports...")
    real_supports = iter_supports_from_dirs(REAL_CAM_DIRS, rng)
    fractal_supports = iter_supports_from_dirs(FRACTAL_CAM_DIRS, rng)

    # Match the number of generated simple shapes to the fractal count for balance.
    n_simple = len(fractal_supports) if fractal_supports else (N_PER_GROUP or 300)

    print("Generating simple-shape supports...")
    groups = {
        "real": real_supports,
        "fractal": fractal_supports,
        "ellipse": gen_simple_shapes(gen_ellipse_support, n_simple, rng),
        "triangle": gen_simple_shapes(gen_triangle_support, n_simple, rng),
        "quadrilateral": gen_simple_shapes(gen_quadrilateral_support, n_simple, rng),
        "polygon": gen_simple_shapes(gen_polygon_support, n_simple, rng),
    }

    print("Computing descriptors...")
    frames = [collect_group(name, sup) for name, sup in groups.items() if sup]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote per-CAM descriptors -> {OUT_CSV}")

    # Summary table (mean +/- std per group x descriptor)
    order = [g for g in ["real", "fractal", "ellipse", "triangle", "quadrilateral", "polygon"]
             if g in df["group"].unique()]
    print("\n=== Summary (mean +/- std) ===")
    summary_rows = []
    for desc in DESCRIPTORS:
        line = f"{desc:24s}"
        for g in order:
            vals = df.loc[df["group"] == g, desc].dropna()
            line += f" | {g}: {vals.mean():.3f}+/-{vals.std():.3f}"
            summary_rows.append({"descriptor": desc, "group": g,
                                 "mean": vals.mean(), "std": vals.std()})
        print(line)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "cam_shape_summary.csv", index=False)

    # Distance of each artificial group's mean to the real mean (lower = closer).
    print("\n=== Normalized distance to REAL mean (lower = more real-like) ===")
    real_mean = {d: df.loc[df["group"] == "real", d].mean() for d in DESCRIPTORS}
    real_std = {d: df.loc[df["group"] == "real", d].std() + 1e-9 for d in DESCRIPTORS}
    for g in order:
        if g == "real":
            continue
        z = []
        for d in DESCRIPTORS:
            gm = df.loc[df["group"] == g, d].mean()
            z.append(abs(gm - real_mean[d]) / real_std[d])
        print(f"  {g:14s}: mean |z| = {np.mean(z):.3f}")

    # Plots: one box per descriptor.
    fig, axes = plt.subplots(1, len(DESCRIPTORS), figsize=(4 * len(DESCRIPTORS), 4))
    for ax, desc in zip(axes, DESCRIPTORS):
        data = [df.loc[df["group"] == g, desc].dropna().values for g in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(desc, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\nWrote plot -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
