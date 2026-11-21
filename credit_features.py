"""Deterministic loading, cleaning, feature engineering, and preprocessors
for the UCI Default of Credit Card Clients dataset."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Order matches the raw columns: index 0 = September ... index 5 = April
MONTHS = ['SEP', 'AUG', 'JUL', 'JUN', 'MAY', 'APR']

RENAME = {'default payment next month': 'default'}
for i, m in enumerate(MONTHS):
    RENAME['PAY_0' if i == 0 else f'PAY_{i + 1}'] = f'PAY_{m}'
    RENAME[f'BILL_AMT{i + 1}'] = f'BILL_{m}'
    RENAME[f'PAY_AMT{i + 1}'] = f'PAYAMT_{m}'

LATE_COLS = [f'LATE_{m}' for m in MONTHS]
STATUS_COLS = [f'STATUS_{m}' for m in MONTHS]
BILL_COLS = [f'BILL_{m}' for m in MONTHS]
PAYAMT_COLS = [f'PAYAMT_{m}' for m in MONTHS]
CAT_COLS = ['EDUCATION', 'MARRIAGE'] + STATUS_COLS
NUM_COLS = ['LIMIT_BAL', 'AGE', 'FEMALE'] + LATE_COLS + BILL_COLS + PAYAMT_COLS

EDU_MAP = {1: 'grad_school', 2: 'university', 3: 'high_school', 4: 'other'}
MAR_MAP = {1: 'married', 2: 'single', 3: 'other'}


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path):
    df = pd.read_excel(path, header=1, index_col='ID')
    return df.rename(columns=RENAME)


def clean(df):
    """Drop exact duplicate records (ID is the index, so it is ignored),
    then fold undocumented codes into each feature's 'other' category."""
    df = df[~df.duplicated(keep='first')].copy()
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    df['FEMALE'] = (df['SEX'] == 2).astype(int)  # docs: 1 = male, 2 = female
    return df.drop(columns='SEX')


def engineer(df):
    """Split each repayment status code into months late + a behavior category."""
    out = df.copy()
    for m in MONTHS:
        code = out.pop(f'PAY_{m}')
        out[f'LATE_{m}'] = code.clip(lower=0)
        out[f'STATUS_{m}'] = np.select(
            [code == -2, code == -1, code == 0],
            ['undocumented', 'paid_full', 'revolving'],
            default='late',
        )
    out['EDUCATION'] = out['EDUCATION'].map(EDU_MAP)
    out['MARRIAGE'] = out['MARRIAGE'].map(MAR_MAP)
    return out


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _log_scale(func):
    return Pipeline([
        ('log', FunctionTransformer(func, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ])


def make_lr_preprocessor():
    return ColumnTransformer([
        ('log', _log_scale(np.log1p), ['LIMIT_BAL'] + PAYAMT_COLS),
        ('slog', _log_scale(signed_log1p), BILL_COLS),
        ('scale', StandardScaler(), ['AGE'] + LATE_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('bin', 'passthrough', ['FEMALE']),
    ], verbose_feature_names_out=False)


def make_tree_preprocessor():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('num', 'passthrough', NUM_COLS),
    ], verbose_feature_names_out=False)
