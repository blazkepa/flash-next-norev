"""MMLU accuracy (greedy by default) against an OpenAI-compatible endpoint, on a fixed 570-question subset of the test
split (data/mmlu_test_570.jsonl: 10 questions from each of the 57 subjects, seed 7). Used as a *general-capability*
guard for weight edits that target coding behaviour. Zero-shot; the model is asked to answer with a single letter and
the first standalone A-D letter in the reply is taken as the answer (no logprobs needed, works with any server).
Usage:
  python3 -I eval/mmlu/run_mmlu.py --model <id> [--limit N] [--workers 4] [--no-think] [--max-tokens 64] [--tag name]"""
import json, os, sys, re, argparse, concurrent.futures as cf, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.llm import chat
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--no-think', action='store_true'); ap.add_argument('--tag', default='')
ap.add_argument('--max-tokens', type=int, default=64); ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--retries', type=int, default=3, help='re-ask after an API/HTTP error')
a = ap.parse_args()
probs = [json.loads(l) for l in open(f'{HERE}/data/mmlu_test_570.jsonl')]
if a.limit: probs = probs[:a.limit]
SYS = "Answer the multiple-choice question. Reply with the letter of the correct answer only (A, B, C or D), nothing else."
def fmt(p):
    opts = '\n'.join(f'{l}. {c}' for l, c in zip('ABCD', p['choices']))
    return f"Subject: {p['subject'].replace('_', ' ')}\n\n{p['question']}\n\n{opts}\n\nAnswer:"
def extract(text):
    m = re.search(r'\b([ABCD])\b', text.strip())
    return m.group(1) if m else ''
def run_one(p):
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_think else None
    err = None
    for attempt in range(a.retries + 1):
        try:
            r = chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': fmt(p)}], a.model,
                     max_tokens=a.max_tokens, temperature=a.temperature, extra=extra); break
        except Exception as e:
            err = e
            if attempt < a.retries: time.sleep(30 * (attempt + 1))
    else:
        return dict(id=p['id'], subject=p['subject'], passed=False, pred='', gold=p['answer'], error=f'api: {err}', seconds=0, tokens=0)
    pred = extract(r['text'])
    return dict(id=p['id'], subject=p['subject'], passed=pred == p['answer'], pred=pred, gold=p['answer'], error='' if pred else 'no letter',
                seconds=r['seconds'], tokens=r['usage'].get('completion_tokens', 0), text=r['text'][:200])
t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex: results = list(ex.map(run_one, probs))
n = len(results); k = sum(r['passed'] for r in results); gen_s = sum(r['seconds'] for r in results); toks = sum(r['tokens'] for r in results)
noletter = sum(1 for r in results if r['error'] == 'no letter'); api = sum(1 for r in results if r['error'].startswith('api'))
by_subj = collections.defaultdict(lambda: [0, 0])
for r in results: by_subj[r['subject']][0] += r['passed']; by_subj[r['subject']][1] += 1
tag = a.tag or re.sub(r'[^A-Za-z0-9._-]', '_', a.model)
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(dict(model=a.model, no_think=a.no_think, max_tokens=a.max_tokens, temperature=a.temperature, n=n, passed=k, accuracy=round(k / n, 4),
               no_letter=noletter, api_errors=api, wall_seconds=round(time.time() - t0, 1), completion_tokens=toks,
               by_subject={s: f'{v[0]}/{v[1]}' for s, v in sorted(by_subj.items())}, results=results),
          open(f'{HERE}/results/{tag}.json', 'w'), indent=1)
print(f"{a.model}: accuracy = {k}/{n} = {k/n:.3f}   no letter {noletter}   api errors {api}   wall {time.time()-t0:.0f}s   tokens {toks}")
