"""Checks that must pass before the model reaches the Space."""
import copy
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import predictor as pr

FIXTURES = json.loads((Path(__file__).parent / 'fixtures.json').read_text())


@pytest.fixture(scope='module')
def card():
    return pr.load()[1]


def good_record():
    return copy.deepcopy(FIXTURES['records'][0])


def test_artifact_is_the_registered_model(card):
    assert pr.sha256_of(pr.MODEL_PATH) == card['sha256']


@pytest.mark.parametrize('package', ['xgboost', 'scikit-learn', 'pandas', 'numpy'])
def test_library_versions_match_training(card, package):
    # A pickled model is only guaranteed to behave the same under the
    # library versions it was saved with.
    assert version(package) == card['versions'][package]


def test_reproduces_module8_predictions():
    got = pr.raw_scores(FIXTURES['records'])
    want = np.array(FIXTURES['expected_raw'])
    assert np.abs(got - want).max() < 1e-6


def test_fixture_auc_matches_training_run():
    from sklearn.metrics import roc_auc_score
    got = roc_auc_score(FIXTURES['y_true'], pr.raw_scores(FIXTURES['records']))
    assert got == pytest.approx(FIXTURES['expected_auc'], abs=1e-9)


def test_decision_follows_cutoff(card):
    res = pr.score(FIXTURES['records'])
    raw = np.array([r['raw_score'] for r in res])
    flags = np.array([r['flagged'] for r in res])
    assert (flags == (raw >= card['cutoff_raw'])).all()
    assert flags.any() and not flags.all()


def test_corrected_probability_is_calibrated_scale(card):
    w = card['class_weight']
    assert pr.corrected_probability(0.5, w) == pytest.approx(1 / (1 + w))
    assert pr.corrected_probability(card['cutoff_raw'], w) == pytest.approx(card['cutoff_corrected'])
    p = pr.corrected_probability(np.linspace(0.01, 0.99, 50), w)
    assert (np.diff(p) > 0).all() and (p > 0).all() and (p < 1).all()


def test_duplicate_records_are_all_scored():
    # cf.clean() drops duplicates during training; scoring must not.
    rec = good_record()
    assert len(pr.score([rec, rec, rec])) == 3


def test_batch_and_single_agree():
    recs = FIXTURES['records'][:5]
    batch = [r['raw_score'] for r in pr.score(recs)]
    single = [pr.score(r)[0]['raw_score'] for r in recs]
    assert np.allclose(batch, single)


@pytest.mark.parametrize('field,value', [
    ('SEX', 3), ('PAY_0', 9), ('PAY_2', -3), ('PAY_3', 1.5), ('AGE', 12),
    ('LIMIT_BAL', 0), ('PAY_AMT1', -5), ('BILL_AMT1', 'abc'), ('MARRIAGE', None),
    ('EDUCATION', float('nan')), ('LIMIT_BAL', True),
])
def test_rejects_bad_values(field, value):
    rec = good_record()
    rec[field] = value
    with pytest.raises(pr.InputError):
        pr.score(rec)


def test_rejects_missing_and_unknown_fields():
    rec = good_record()
    del rec['AGE']
    with pytest.raises(pr.InputError, match='missing'):
        pr.score(rec)
    rec = good_record()
    rec['INCOME'] = 1
    with pytest.raises(pr.InputError, match='unknown'):
        pr.score(rec)


def test_rejects_empty_and_oversized_batches():
    with pytest.raises(pr.InputError):
        pr.score([])
    with pytest.raises(pr.InputError):
        pr.score([good_record()] * 1001)


def test_warns_outside_training_range():
    rec = good_record()
    rec['AGE'] = 90
    res = pr.score(rec)[0]
    assert res['warnings'] and 'AGE' in res['warnings'][0]
    assert not pr.score(good_record())[0]['warnings']


def test_undocumented_codes_fold_like_training():
    # EDUCATION 0, 5 and 6 are all "other" (4) in training; MARRIAGE 0 is 3.
    rec = good_record()
    scores = []
    for edu in (4, 0, 5, 6):
        rec['EDUCATION'] = edu
        scores.append(pr.score(rec)[0]['raw_score'])
    assert len(set(scores)) == 1
