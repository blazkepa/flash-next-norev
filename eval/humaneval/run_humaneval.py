"""HumanEval pass@1 (greedy) against an OpenAI-compatible endpoint. Executes generated code in
subprocesses with a timeout (python3 -I, temp dir). Usage:
  python3 -I eval/humaneval/run_humaneval.py --model <id> [--limit N] [--workers 4] [--no-think] [--tag name]"""
import json, os, sys, re, argparse, subprocess, tempfile, concurrent.futures as cf, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.llm import chat
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--no-think', action='store_true'); ap.add_argument('--tag', default='')
ap.add_argument('--max-tokens', type=int, default=1024); a = ap.parse_args()
probs = [json.loads(l) for l in open(f'{HERE}/data/HumanEval.jsonl')]
if a.limit: probs = probs[:a.limit]
SYS = "You are a Python programmer. Complete the function. Reply with ONLY the complete Python code in one ```python block, no explanations."
def extract(text):
    m = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    if m: return m[-1]
    m = re.search(r'```(?:python)?\n(.*)$', text, re.S)  # unclosed fence
    return m.group(1) if m else text
def run_one(p):
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_think else None
    try:
        r = chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': p['prompt']}], a.model, max_tokens=a.max_tokens, extra=extra)
    except Exception as e:
        return dict(task_id=p['task_id'], passed=False, error=f'api: {e}', seconds=0, tokens=0)
    code = extract(r['text'])
    # if model returned only the body, prepend the prompt (signature + docstring)
    if f"def {p['entry_point']}" not in code: code = p['prompt'] + code
    prog = code + '\n\n' + p['test'] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'prog.py'); open(path, 'w').write(prog)
        try:
            res = subprocess.run([sys.executable, '-I', path], cwd=d, capture_output=True, text=True, timeout=10)
            ok, err = res.returncode == 0, res.stderr[-300:]
        except subprocess.TimeoutExpired: ok, err = False, 'timeout'
    return dict(task_id=p['task_id'], passed=ok, error='' if ok else err, seconds=r['seconds'], tokens=r['usage'].get('completion_tokens', 0), code=code)
t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex: results = list(ex.map(run_one, probs))
n = len(results); k = sum(r['passed'] for r in results); gen_s = sum(r['seconds'] for r in results); toks = sum(r['tokens'] for r in results)
tag = a.tag or re.sub(r'[^A-Za-z0-9._-]', '_', a.model)
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(dict(model=a.model, no_think=a.no_think, n=n, passed=k, pass_at_1=round(k / n, 4), wall_seconds=round(time.time() - t0, 1),
               completion_tokens=toks, tok_per_s_per_stream=round(toks / gen_s, 1) if gen_s else None, results=results),
          open(f'{HERE}/results/{tag}.json', 'w'), indent=1)
print(f"{a.model}: pass@1 = {k}/{n} = {k/n:.3f}   wall {time.time()-t0:.0f}s   tokens {toks}   ~{toks/gen_s if gen_s else 0:.0f} tok/s per stream")
errs = [r for r in results if not r['passed']][:5]
for r in errs: print(' FAIL', r['task_id'], (r['error'] or '')[-120:].replace('\n', ' | '))
