"""End-to-end: generate raw data -> clean/validate -> train Random Forest.

Run:  python -m src.pipeline [--regenerate]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import generate_data
from .cleaning import clean
from .model import train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RAW = DATA / "raw_flights.csv"
CLEAN = DATA / "clean_flights.parquet"
LOG = DATA / "cleaning_log.csv"


def run(regenerate: bool = False) -> None:
    if regenerate or not RAW.exists():
        generate_data.main(RAW)
    raw = pd.read_csv(RAW, dtype={"Previous_Flight": "string"})
    df, log = clean(raw)
    df.to_parquet(CLEAN, index=False)
    log.to_csv(LOG, index=False)
    print(log.to_string(index=False))
    m = train(df, MODELS)
    print(f"\nRandom Forest  best={m['best_params']}  CV AUC={m['cv_auc']:.3f}")
    print(f"Test: acc={m['accuracy']:.3f} (baseline {m['baseline_accuracy']:.3f})  "
          f"prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} "
          f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")


if __name__ == "__main__":
    run(regenerate="--regenerate" in sys.argv)
