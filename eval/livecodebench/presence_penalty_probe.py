"""Check that presence_penalty takes effect in each oMLX serving path, for one or more models.
Greedy decoding, 400 tokens, the first 4 hard tasks of the subset; for each model and each concurrency (1 request at a time,
then 4 at once) it asks with penalty 0 twice (determinism control), 1.5 and 100, and prints the first character at which each
answer departs from the first penalty-0 answer (-1 = identical). The 4-at-once round is what puts r6s05 on oMLX's batched path.
  python3 -I eval/livecodebench/presence_penalty_probe.py <model id> [<model id> ...]
Writes the raw answers next to the script as presence_penalty_probe.json."""
import sys, os, json, time, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_livecodebench as R
models = sys.argv[1:]
sub, probs = R.load_problems(100)
pick = [p for p in probs if p['difficulty'] == 'hard'][:4]
def ask(model, p, pen):
    extra = {'chat_template_kwargs': {'enable_thinking': False}}
    if pen: extra['presence_penalty'] = pen
    r = R.chat([{'role': 'system', 'content': R.SYS}, {'role': 'user', 'content': R.build_prompt(p)}], model, max_tokens=400, temperature=0.0, extra=extra, timeout=900)
    return r['text']
def div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return -1 if len(a) == len(b) else n
out = {}
for model in models:
    for mode, workers in (('single', 1), ('conc4', 4)):
        res = {}
        for label, pen in (('p0a', 0), ('p0b', 0), ('p1.5', 1.5), ('p100', 100)):
            t = time.time()
            with cf.ThreadPoolExecutor(workers) as ex: res[label] = list(ex.map(lambda p: ask(model, p, pen), pick))
            print(f"== {model} {mode} {label} {time.strftime('%H:%M:%S')} ({time.time()-t:.0f}s)", flush=True)
        for label in ('p0b', 'p1.5', 'p100'):
            d = [div(res['p0a'][i], res[label][i]) for i in range(len(pick))]
            print(f"   {mode:6} {label:5} first divergence vs p0a (chars, -1 = identical): {d}  lens {[len(x) for x in res[label]]}", flush=True)
        out[f'{model}|{mode}'] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presence_penalty_probe.json'), 'w'), indent=1)
