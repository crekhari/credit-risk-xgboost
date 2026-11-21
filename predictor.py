"""Load the tuned XGBoost model and score clients.

Inputs use the raw UCI field names and codes (Yeh, 2009), so a record can be
copied straight from the original dataset. The same cleaning and feature
engineering as training come from credit_features.py.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import credit_features as cf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model' / 'xgb_tuned.joblib'
CARD_PATH = ROOT / 'model' / 'model_card.json'

PAY_FIELDS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_FIELDS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAYAMT_FIELDS = [f'PAY_AMT{i}' for i in range(1, 7)]
FIELDS = (['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
          + PAY_FIELDS + BILL_FIELDS + PAYAMT_FIELDS)

# Hard limits: inputs outside these sets or ranges are rejected.
ALLOWED_CODES = {
    'SEX': {1, 2},
    'EDUCATION': {0, 1, 2, 3, 4, 5, 6},
    'MARRIAGE': {0, 1, 2, 3},
    **{f: set(range(-2, 9)) for f in PAY_FIELDS},
}
# Soft limits: the training data's range. Outside it the model extrapolates,
# so the result carries a warning instead of an error.
TRAINING_RANGE = {'LIMIT_BAL': (10_000, 1_000_000), 'AGE': (21, 79)}


class InputError(ValueError):
    """A record that cannot be scored."""


@lru_cache(maxsize=1)
def load():
    """Return the model bundle and model card. Cached, so it loads once."""
    bundle = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text())
    return bundle, card


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_number(field, value):
    if isinstance(value, bool) or value is None:
        raise InputError(f'{field} must be a number, got {value!r}')
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError(f'{field} must be a number, got {value!r}') from None
    if not np.isfinite(x):
        raise InputError(f'{field} must be finite, got {value!r}')
    return x


def validate(record: dict) -> tuple[dict, list[str]]:
    """Check one record. Return clean values and any warnings."""
    if not isinstance(record, dict):
        raise InputError('each record must be a JSON object')
    missing = [f for f in FIELDS if f not in record]
    if missing:
        raise InputError(f'missing fields: {", ".join(missing)}')
    extra = sorted(set(record) - set(FIELDS))
    if extra:
        raise InputError(f'unknown fields: {", ".join(extra)}')

    out, warnings = {}, []
    for f in FIELDS:
        x = _as_number(f, record[f])
        if f in ALLOWED_CODES:
            if x != int(x) or int(x) not in ALLOWED_CODES[f]:
                raise InputError(f'{f} must be one of {sorted(ALLOWED_CODES[f])}, got {record[f]!r}')
            x = int(x)
        out[f] = x
    if out['LIMIT_BAL'] <= 0:
        raise InputError('LIMIT_BAL must be positive')
    if not 18 <= out['AGE'] <= 100:
        raise InputError('AGE must be between 18 and 100')
    if any(out[f] < 0 for f in PAYAMT_FIELDS):
        raise InputError('PAY_AMT fields cannot be negative')
    for f, (lo, hi) in TRAINING_RANGE.items():
        if not lo <= out[f] <= hi:
            warnings.append(f'{f} = {out[f]:,.0f} is outside the training range '
                            f'{lo:,}-{hi:,}, so this score is an extrapolation.')
    return out, warnings


def to_features(records: list[dict]) -> pd.DataFrame:
    """Raw UCI records to the 29 model features.

    cf.clean() also drops duplicate rows, which is right for training but
    wrong for scoring, so the code folding is repeated here without it.
    """
    df = pd.DataFrame(records, columns=FIELDS).rename(columns=cf.RENAME)
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    df['FEMALE'] = (df['SEX'] == 2).astype(int)
    df = cf.engineer(df.drop(columns='SEX'))
    return df[cf.CAT_COLS + cf.NUM_COLS]


def corrected_probability(raw, weight):
    """Undo the class weight (prior correction, King & Zeng, 2001).

    Weighting defaulters by w multiplies the model's odds by w, so dividing
    the odds by w recovers a calibrated probability.
    """
    raw = np.asarray(raw, dtype=float)
    return raw / (raw + weight * (1 - raw))


def raw_scores(records: list[dict]) -> np.ndarray:
    bundle, _ = load()
    Z = bundle['prep'].transform(to_features(records))
    return bundle['model'].predict_proba(Z, iteration_range=(0, bundle['n_trees']))[:, 1]


def score(records: list[dict] | dict) -> list[dict]:
    """Validate and score one record or a list of records."""
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise InputError('send one record or a non-empty list of records')
    if len(records) > 1000:
        raise InputError('send at most 1,000 records per request')
    checked = [validate(r) for r in records]
    _, card = load()
    raw = raw_scores([c[0] for c in checked])
    prob = corrected_probability(raw, card['class_weight'])
    results = []
    for r, p, (_, warnings) in zip(raw, prob, checked):
        flag = bool(r >= card['cutoff_raw'])
        results.append({
            'default_probability': round(float(p), 4),
            'decision': 'flag for review' if flag else 'no flag',
            'flagged': flag,
            'flag_threshold': round(card['cutoff_corrected'], 4),
            'raw_score': float(r),
            'warnings': warnings,
            'model_version': card['sha256'][:12],
        })
    return results
