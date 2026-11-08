"""GSM8K accuracy (greedy by default) against an OpenAI-compatible endpoint, on a fixed 300-problem subset of the
test split (data/gsm8k_test_300.jsonl, 10 % of the 1 319 problems, seed 7). Used as a *general-capability* guard: the
weight edits in this project target coding behaviour and must not move grade-school arithmetic.
The model is asked to end with "#### <number>"; the last number in the answer is compared with the gold number.
Usage:
  python3 -I eval/gsm8k/run_gsm8k.py --model <id> [--limit N] [--workers 4] [--no-think] [--max-tokens 2048] [--tag name]"""
import json, os, sys, re, argparse, concurrent.futures as cf, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.llm import chat
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--no-think', action='store_true'); ap.add_argument('--tag', default='')
ap.add_argument('--max-tokens', type=int, default=2048); ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--retries', type=int, default=3, help='re-ask after an API/HTTP error (a server under memory pressure drops connections)')
a = ap.parse_args()
probs = [json.loads(l) for l in open(f'{HERE}/data/gsm8k_test_300.jsonl')]
if a.limit: probs = probs[:a.limit]
SYS = "Solve the math word problem. Think step by step, then finish with a final line of the form '#### <number>' containing only the final numeric answer."
NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
def extract(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    if m: return m[-1].replace(',', '')
    nums = NUM.findall(text)
    return nums[-1].replace(',', '') if nums else ''
def same(pred, gold):
    try: return abs(float(pred) - float(gold)) < 1e-6
    except ValueError: return pred.strip() == gold.strip()
def run_one(p):
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_think else None
    err = None
    for attempt in range(a.retries + 1):
        try:
            r = chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': p['question']}], a.model,
                     max_tokens=a.max_tokens, temperature=a.temperature, extra=extra); break
        except Exception as e:
            err = e
            if attempt < a.retries: time.sleep(30 * (attempt + 1))
    else:
        return dict(id=p['id'], passed=False, pred='', gold=p['gold'], error=f'api: {err}', seconds=0, tokens=0, truncated=False)
    toks = r['usage'].get('completion_tokens', 0); pred = extract(r['text'])
    return dict(id=p['id'], passed=same(pred, p['gold']), pred=pred, gold=p['gold'], error='', seconds=r['seconds'], tokens=toks,
                truncated=toks >= a.max_tokens, text=r['text'])
t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex: results = list(ex.map(run_one, probs))
n = len(results); k = sum(r['passed'] for r in results); gen_s = sum(r['seconds'] for r in results); toks = sum(r['tokens'] for r in results)
trunc = sum(r['truncated'] for r in results); api = sum(1 for r in results if r['error'].startswith('api'))
tag = a.tag or re.sub(r'[^A-Za-z0-9._-]', '_', a.model)
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(dict(model=a.model, no_think=a.no_think, max_tokens=a.max_tokens, temperature=a.temperature, n=n, passed=k, accuracy=round(k / n, 4),
               truncated=trunc, api_errors=api, wall_seconds=round(time.time() - t0, 1), completion_tokens=toks,
               tok_per_s_per_stream=round(toks / gen_s, 1) if gen_s else None, results=results),
          open(f'{HERE}/results/{tag}.json', 'w'), indent=1)
print(f"{a.model}: accuracy = {k}/{n} = {k/n:.3f}   truncated {trunc}   api errors {api}   wall {time.time()-t0:.0f}s   tokens {toks}   ~{toks/gen_s if gen_s else 0:.0f} tok/s per stream")
for r in [r for r in results if not r['passed']][:5]: print(' FAIL', r['id'], 'pred', r['pred'], 'gold', r['gold'], (r['error'] or '')[-80:])
