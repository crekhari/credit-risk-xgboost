"""The Gradio app builds and its two handlers return the right shapes."""
import json
from pathlib import Path

import gradio as gr
import pytest

import app
import predictor as pr

FIXTURES = json.loads((Path(__file__).parent / 'fixtures.json').read_text())


def test_app_builds():
    assert isinstance(app.demo, gr.Blocks)


def test_examples_are_valid_inputs():
    for ex in app.CARD['examples']:
        pr.validate(ex)
        assert ex['EDUCATION'] in dict((v, k) for k, v in app.EDU_CHOICES)
        assert ex['MARRIAGE'] in dict((v, k) for k, v in app.MAR_CHOICES)


def test_score_form_returns_summary_and_json():
    rec = FIXTURES['records'][0]
    summary, res = app.score_form(*[rec[f] for f in pr.FIELDS])
    assert 'chance of default' in summary
    assert res['raw_score'] == pytest.approx(FIXTURES['expected_raw'][0], abs=1e-6)


def test_score_form_shows_input_errors():
    rec = dict(FIXTURES['records'][0], SEX=3)
    with pytest.raises(gr.Error):
        app.score_form(*[rec[f] for f in pr.FIELDS])


def test_score_json_single_and_batch():
    recs = FIXTURES['records'][:3]
    one = json.loads(app.score_json(json.dumps(recs[0])))
    many = json.loads(app.score_json(json.dumps(recs)))
    assert len(one) == 1 and len(many) == 3


def test_score_json_reports_errors_without_crashing():
    assert 'error' in json.loads(app.score_json('{not json'))
    assert 'error' in json.loads(app.score_json(json.dumps({'AGE': 30})))
