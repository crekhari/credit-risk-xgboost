"""After a deploy: wait for the Space to run the new commit, then call it.

Exits non-zero if the Space fails to build, never starts, or returns scores
that differ from the fixtures.
"""
import json
import os
import sys
import time
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi

SPACE_ID = os.environ['SPACE_ID']
TIMEOUT = int(os.environ.get('SMOKE_TIMEOUT', 1200))
FIXTURES = json.loads((Path(__file__).resolve().parents[1] / 'tests' / 'fixtures.json').read_text())
FAILED = {'BUILD_ERROR', 'RUNTIME_ERROR', 'CONFIG_ERROR', 'NO_APP_FILE', 'PAUSED'}

api = HfApi(token=os.environ.get('HF_TOKEN'))
target = api.space_info(SPACE_ID).sha
print(f'waiting for {SPACE_ID} to run commit {target[:8]}')

start, last = time.time(), None
while True:
    rt = api.get_space_runtime(SPACE_ID)
    running_sha = (rt.raw or {}).get('sha')
    state = (rt.stage, running_sha[:8] if running_sha else None, rt.hardware)
    if state != last:
        print(f'{time.time() - start:6.0f}s  stage={rt.stage}  sha={state[1]}  hardware={rt.hardware}')
        last = state
    if rt.stage in FAILED:
        sys.exit(f'Space failed with stage {rt.stage}')
    if rt.stage == 'RUNNING' and (running_sha is None or running_sha == target):
        break
    if time.time() - start > TIMEOUT:
        sys.exit(f'Space not running after {TIMEOUT}s')
    time.sleep(15)
print(f'Space running after {time.time() - start:.0f}s')

client = Client(SPACE_ID, verbose=False)
recs = FIXTURES['records'][:20]
t0 = time.time()
res = json.loads(client.predict(json.dumps(recs), api_name='/score_json'))
latency = time.time() - t0
if isinstance(res, dict) and 'error' in res:
    sys.exit(f'Space returned an error: {res["error"]}')
gap = max(abs(r['raw_score'] - e) for r, e in zip(res, FIXTURES['expected_raw'][:20]))
print(f'20 records scored in {latency:.2f}s | largest gap from fixtures {gap:.2e}')
if len(res) != 20 or gap > 1e-6:
    sys.exit('live Space does not reproduce the fixtures')
print('smoke test passed')
