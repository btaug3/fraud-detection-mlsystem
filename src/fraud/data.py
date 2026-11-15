"""Ingestion and leakage-free temporal splitting."""
import logging

import pandas as pd

from .config import DATA_RAW, ID_COL, TARGET, TIME_COL, TRAIN_FRAC, VAL_FRAC

log = logging.getLogger(__name__)


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)          # 1) remove exact duplicates BEFORE splitting
    log.info("Loaded %d rows, dropped %d exact duplicates", n0, n0 - len(df))
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)   # 2) strict time order (stable sort)
    df.insert(0, ID_COL, [f"txn_{i:07d}" for i in range(len(df))])          # 3) record IDs for outputs
    return df


def temporal_split(df: pd.DataFrame):
    """Every val/test transaction happens AFTER every training transaction (no look-ahead)."""
    t_train = df[TIME_COL].quantile(TRAIN_FRAC)
    t_val = df[TIME_COL].quantile(TRAIN_FRAC + VAL_FRAC)
    train = df[df[TIME_COL] < t_train]
    val = df[(df[TIME_COL] >= t_train) & (df[TIME_COL] < t_val)]
    test = df[df[TIME_COL] >= t_val]

    # hard guarantees - the program crashes if the split ever overlaps in time
    assert train[TIME_COL].max() < val[TIME_COL].min()
    assert val[TIME_COL].max() < test[TIME_COL].min()

    for name, part in [("train", train), ("val", val), ("test", test)]:
        log.info("%-5s n=%6d fraud=%4d rate=%.4f%%", name, len(part),
                 part[TARGET].sum(), 100 * part[TARGET].mean())
    cutoffs = {"train_end_s": float(t_train), "val_end_s": float(t_val)}
    return train, val, test, cutoffs