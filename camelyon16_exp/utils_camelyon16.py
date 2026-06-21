"""Shared loaders / split logic for the Camelyon16 scripts."""
from pathlib import Path
import numpy as np
import pandas as pd


CAMELYON16_ROOT = '/mnt/d/CAMELYON16'
CSV_PATH = f'{CAMELYON16_ROOT}/camelyon16_patch_data.csv'

SLIDES_PER_CLASS_TRAIN = 50
RANDOM_SEED = 2024


def load_camelyon16_df() -> pd.DataFrame:
    """Read the CSV and prepend CAMELYON16_ROOT to img_path."""
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['img_path'] = CAMELYON16_ROOT + '/' + df['img_path']
    return df


def sample_train_slides(
    df: pd.DataFrame,
    n_per_class: int = SLIDES_PER_CLASS_TRAIN,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Pick `n_per_class` slide names from train, balanced by slide_label."""
    rng = np.random.RandomState(seed)
    train = df[df['subset'] == 'train']
    slides = train.drop_duplicates('slide_name')[['slide_name', 'slide_label']]
    picked: list[str] = []
    for label in [0, 1]:
        pool = slides[slides['slide_label'] == label]['slide_name'].tolist()
        if n_per_class > len(pool):
            print(
                f"[warn] requested {n_per_class} train slides for slide_label={label}, "
                f"only {len(pool)} available"
            )
            picked.extend(pool)
        else:
            idx = rng.choice(len(pool), size=n_per_class, replace=False)
            picked.extend([pool[i] for i in idx])
    return sorted(picked)


def get_split_dfs(
    df: pd.DataFrame,
    n_per_class: int = SLIDES_PER_CLASS_TRAIN,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} dataframes.

    `train` is filtered to the sampled `n_per_class * 2` slides. `val` and
    `test` are returned unchanged from the CSV.
    """
    train_slides = sample_train_slides(df, n_per_class=n_per_class, seed=seed)
    return {
        'train': df[(df['subset'] == 'train') & (df['slide_name'].isin(train_slides))]
            .reset_index(drop=True),
        'val': df[df['subset'] == 'val'].reset_index(drop=True),
        'test': df[df['subset'] == 'test'].reset_index(drop=True),
    }
