---
title: Credit Default Risk (XGBoost)
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Tuned XGBoost credit card default model (AAI 6610)
---

# Credit card default risk

Tuned XGBoost model from my AAI 6610 Applied Machine Learning project. It scores the chance
that a credit card client defaults next month, using the UCI Default of Credit Card Clients
data (Yeh, 2009).

* Test AUC 0.7868, 327 trees, flag threshold 27.1%.
* Code: https://github.com/crekhari/credit-risk-xgboost. Every push to `main` runs the tests, syncs this Space,
  then calls the live app to check it reproduces the fixtures.
* Course demo only. Not for real credit decisions.

## Call it from Python

```python
from gradio_client import Client
import json
client = Client("chiraag/credit-risk-xgboost")
record = {"LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 46, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 50297, "BILL_AMT2": 46613, "BILL_AMT3": 46666, "BILL_AMT4": 20162, "BILL_AMT5": 19864, "BILL_AMT6": 19680, "PAY_AMT1": 1852, "PAY_AMT2": 4050, "PAY_AMT3": 676, "PAY_AMT4": 730, "PAY_AMT5": 866, "PAY_AMT6": 3000}
print(client.predict(json.dumps(record), api_name="/score_json"))
```
