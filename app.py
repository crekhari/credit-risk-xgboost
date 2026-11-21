"""Gradio front end for the tuned XGBoost credit default model."""
import json

import gradio as gr

import predictor as pr

# ZeroGPU Spaces expect at least one @spaces.GPU function at startup. This
# model runs on CPU, so the placeholder is never called and uses no GPU quota.
# On CPU hardware and in CI the package is absent and this block is skipped.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except Exception:  # noqa: BLE001
    pass

BUNDLE, CARD = pr.load()

MONTHS = ['September', 'August', 'July', 'June', 'May', 'April']
STATUS_CHOICES = [('No credit use (-2)', -2), ('Paid in full (-1)', -1),
                  ('Revolving, minimum paid (0)', 0)] + \
                 [(f'{k} month{"s" if k > 1 else ""} late ({k})', k) for k in range(1, 9)]
EDU_CHOICES = [('Graduate school', 1), ('University', 2), ('High school', 3), ('Other', 4)]
MAR_CHOICES = [('Married', 1), ('Single', 2), ('Other', 3)]
SEX_CHOICES = [('Male', 1), ('Female', 2)]

INTRO = f"""
# Credit card default risk

Scores the chance that a credit card client defaults next month. The model is
the tuned XGBoost classifier from my AAI 6610 project ({CARD['n_trees']} trees),
trained on the UCI *Default of Credit Card Clients* data (Taiwan, April to
September 2005).

Test set: AUC {CARD['test_metrics']['auc']:.3f}, precision
{CARD['test_metrics']['precision']:.3f}, recall {CARD['test_metrics']['recall']:.3f}.
A client is flagged when the probability reaches
{CARD['cutoff_corrected']:.1%}, the cutoff that maximized F1 in cross-validation.

**Course demo only.** Do not use it for real credit decisions.
"""


def _form_to_record(*values):
    return dict(zip(pr.FIELDS, values))


def score_form(*values):
    """Score the form. Returns a summary and the full JSON result."""
    try:
        res = pr.score(_form_to_record(*values))[0]
    except pr.InputError as e:
        raise gr.Error(str(e))
    lines = [f"## {res['default_probability']:.1%} chance of default",
             f"**Decision:** {res['decision']} "
             f"(threshold {res['flag_threshold']:.1%})"]
    lines += [f'> {w}' for w in res['warnings']]
    return '\n\n'.join(lines), res


def score_json(payload: str) -> str:
    """API: one record or a list of records as JSON text, in UCI field names."""
    try:
        return json.dumps(pr.score(json.loads(payload)))
    except json.JSONDecodeError as e:
        return json.dumps({'error': f'invalid JSON: {e}'})
    except pr.InputError as e:
        return json.dumps({'error': str(e)})


with gr.Blocks() as demo:
    gr.Markdown(INTRO)
    with gr.Tab('Score a client'):
        with gr.Row():
            limit = gr.Number(label='Credit limit (NT$)', value=150_000, minimum=1)
            sex = gr.Dropdown(SEX_CHOICES, value=2, label='Sex')
            edu = gr.Dropdown(EDU_CHOICES, value=2, label='Education')
            mar = gr.Dropdown(MAR_CHOICES, value=2, label='Marital status')
            age = gr.Number(label='Age', value=35, precision=0)
        gr.Markdown('### Repayment status by month')
        with gr.Row():
            pay = [gr.Dropdown(STATUS_CHOICES, value=0, label=m) for m in MONTHS]
        gr.Markdown('### Bill amount by month (NT$)')
        with gr.Row():
            bill = [gr.Number(label=m, value=40_000) for m in MONTHS]
        gr.Markdown('### Amount paid by month (NT$)')
        with gr.Row():
            paid = [gr.Number(label=m, value=2_000, minimum=0) for m in MONTHS]
        inputs = [limit, sex, edu, mar, age, *pay, *bill, *paid]
        btn = gr.Button('Score', variant='primary')
        summary = gr.Markdown()
        detail = gr.JSON(label='Full result')
        btn.click(score_form, inputs, [summary, detail], api_name='score_form')
        gr.Examples([[ex[f] for f in pr.FIELDS] for ex in CARD['examples']],
                    inputs=inputs, label='Clients from the test set (low, typical, high risk)')
    with gr.Tab('JSON API'):
        gr.Markdown('Send one record or a list of up to 1,000 records, using the UCI '
                    'field names. Call it from Python with `gradio_client` and '
                    '`api_name="/score_json"`.')
        payload = gr.Code(json.dumps(CARD['examples'][0], indent=2), language='json',
                          label='Request')
        out = gr.Code(language='json', label='Response')
        gr.Button('Send').click(score_json, payload, out, api_name='score_json')

if __name__ == '__main__':
    demo.launch()
