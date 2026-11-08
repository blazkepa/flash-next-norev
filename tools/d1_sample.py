#!/usr/bin/env python3
"""Rejection-sampling helper: the model generates k solutions per problem, stdin/stdout tests filter out the wrong ones.

python3 -I tools/d1_sample.py --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://127.0.0.1:8081/v1 --limit 100 --k 4 \
    [--workers 4] [--temperature 0.7] [--top-p 0.8] [--max-tokens 2048] [--min-tests 5] [--max-tests 50] [--timeout 10] [--out data/d1/raw/samples.jsonl]
Teacher mode (HTTP, oMLX): --model Qwen3.8-Flash-Next-oQ5e-mtp --base-url http://127.0.0.1:8000/v1 --no-think --only-failed-in data/d1/raw/samples.norev-s24.jsonl

Input: data/d1/raw/*.parquet (open-r1 verifiable-coding-problems, stdin_stdout tests). Grader and prompt are imported from eval/livecodebench
(same format as the measurement, so we train on exactly what we measure). Output: jsonl, one row per sample; resumable (skips problems already in the file).
Excluded: problems with < min-tests tests and those sharing >= 3 12-grams with the LCB subset (leakage guard for the safeguard set)."""
import json, os, sys, re, argparse, time, threading, collections, random, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, f'{ROOT}/eval'); sys.path.insert(0, f'{ROOT}/eval/livecodebench')
from common.llm import chat
from run_livecodebench import grade, extract, SYS, FMT_STDIN
import pyarrow.parquet as pq

PARQUET = f'{ROOT}/data/d1/raw/open-r1_vcp_decontaminated-tested-shuffled.parquet'
LCB = f'{ROOT}/eval/livecodebench/data/test6.jsonl'

def norm(s): return re.sub(r'\W+', ' ', s.lower()).strip()
def grams(s, n=12):
    w = norm(s).split(); return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}

def build_prompt(statement):
    s = re.sub(r'^Solve the following coding problem using the programming language python:\s*', '', statement.strip())
    return f"### Question:\n{s}\n\n### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n"

def load_problems(limit, min_tests, max_tests, seed):
    rows = pq.read_table(PARQUET).to_pylist()
    lcb_grams = set().union(*[grams(json.loads(l)['question_content']) for l in open(LCB)])
    out = []; dropped = collections.Counter()
    for r in rows:
        tc = r['verification_info']['test_cases']
        if len(tc) < min_tests: dropped['too few tests'] += 1; continue
        if len(grams(r['problem']) & lcb_grams) >= 3: dropped['similar to LCB'] += 1; continue
        tests = [dict(input=c['input'], output=c['output']) for c in tc]
        if max_tests and len(tests) > max_tests: random.Random(seed).shuffle(tests); tests = tests[:max_tests]
        out.append(dict(id=r['problem_id'], source=r['source'], difficulty=(r['metadata'] or {}).get('difficulty'), statement=r['problem'],
                        tests=tests, tests_total=len(tc), func_name=None))
        if limit and len(out) >= limit: break
    return out, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--base-url', default='http://127.0.0.1:8081/v1')
    ap.add_argument('--limit', type=int, default=100); ap.add_argument('--k', type=int, default=4); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.7); ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048); ap.add_argument('--min-tests', type=int, default=5); ap.add_argument('--max-tests', type=int, default=50)
    ap.add_argument('--timeout', type=float, default=10.0); ap.add_argument('--mem-mb', type=int, default=1024); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=f'{ROOT}/data/d1/raw/samples.jsonl')
    ap.add_argument('--local', action='store_true', help='generate in-process via mlx_lm.batch_generate (no server); --model = path/HF id')
    ap.add_argument('--batch', type=int, default=16, help='number of sequences per batch (--local only)')
    ap.add_argument('--adapter', default=None, help='LoRA adapter directory loaded in-process (--local only)')
    ap.add_argument('--skip', type=int, default=0, help='skip the first N problems after filtering (e.g. held-out set: --skip 1000 --limit 100)')
    ap.add_argument('--only-failed-in', default=None, help='only problems whose samples in this jsonl ALL failed the tests (teacher mode: the student\'s 0/k problems)')
    ap.add_argument('--sys-suffix', default='', help='addendum to the system prompt (HTTP only; e.g. for the teacher: reason briefly, then code)')
    ap.add_argument('--max-problems', type=int, default=0, help='after all filters keep only the first N problems (trial run)')
    ap.add_argument('--no-think', action='store_true', help='disable thinking (chat_template_kwargs.enable_thinking=false; HTTP only, Qwen3.x models on oMLX)')
    a = ap.parse_args(); mem = a.mem_mb * 1024 * 1024
    if a.local: return main_local(a, mem)
    probs, dropped = load_problems(a.skip + a.limit if a.limit else 0, a.min_tests, a.max_tests, a.seed); probs = probs[a.skip:]
    if a.only_failed_in:
        passed = collections.defaultdict(bool)
        for l in open(a.only_failed_in): r = json.loads(l); passed[r['id']] = passed[r['id']] or r['passed']
        probs = [p for p in probs if p['id'] in passed and not passed[p['id']]]
        print(f"teacher mode: {len(probs)} problems with 0/k in {a.only_failed_in}", flush=True)
    if a.max_problems: probs = probs[:a.max_problems]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)['id'])
    todo = [p for p in probs if p['id'] not in done]
    print(f"problems {len(probs)} (rejected at selection: {dict(dropped)}), done {len(done)}, to do {len(todo)}, k={a.k}, model {a.model}", flush=True)
    lock = threading.Lock(); t0 = time.time(); stats = collections.Counter(); toks = [0]; gen_s = [0.0]; n_done = [0]
    def one(p, i):
        msgs = [{'role': 'system', 'content': SYS + ('\n' + a.sys_suffix if a.sys_suffix else '')}, {'role': 'user', 'content': build_prompt(p['statement'])}]
        try: r = chat(msgs, a.model, base_url=a.base_url, max_tokens=a.max_tokens, temperature=a.temperature, extra={'top_p': a.top_p, 'seed': a.seed * 1000 + i, **({'chat_template_kwargs': {'enable_thinking': False}} if a.no_think else {})})
        except Exception as e: return dict(id=p['id'], sample=i, error=f'api: {e}', passed=False, tokens=0, seconds=0, truncated=False, code='', text='', tests_run=0, tests_total=len(p['tests']))
        code = extract(r['text']); tok = r['usage'].get('completion_tokens', 0); g = grade(code, p, a.timeout, mem)
        return dict(id=p['id'], sample=i, source=p['source'], difficulty=p['difficulty'], **g, tokens=tok, seconds=r['seconds'], truncated=tok >= a.max_tokens, code=code, text=r['text'])
    def run_problem(p):
        res = [one(p, i) for i in range(a.k)]
        with lock:
            with open(a.out, 'a') as f:
                for r in res: f.write(json.dumps(r, ensure_ascii=False) + '\n')
            k_pass = sum(r['passed'] for r in res); stats[k_pass] += 1; toks[0] += sum(r['tokens'] for r in res); gen_s[0] += sum(r['seconds'] for r in res); n_done[0] += 1
            el = time.time() - t0; eta = el / n_done[0] * (len(todo) - n_done[0])
            print(f"  {n_done[0]:>4}/{len(todo)} {p['id']:<12} {str(p['difficulty']):<16} pass {k_pass}/{a.k}  tok {[r['tokens'] for r in res]}  "
                  f"{el/60:5.1f} min, ETA {eta/60:4.0f} min  {res[0]['error'][:70] if not res[0]['passed'] else ''}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex: list(ex.map(run_problem, todo))
    wall = time.time() - t0
    print(f"done: {n_done[0]} problems in {wall/60:.1f} min, tokens {toks[0]} (~{toks[0]/wall:.0f} tok/s total), distribution of passed samples out of {a.k}: {dict(sorted(stats.items()))} -> {a.out}")

def main_local(a, mem):
    """Same protocol as main(), but batched in-process generation: batch = (batch // k) problems × k samples."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    probs, dropped = load_problems(a.skip + a.limit if a.limit else 0, a.min_tests, a.max_tests, a.seed); probs = probs[a.skip:]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)['id'])
    todo = [p for p in probs if p['id'] not in done]
    print(f"problems {len(probs)} (skip {a.skip}; rejected at selection: {dict(dropped)}), done {len(done)}, to do {len(todo)}, k={a.k}, model {a.model} adapter {a.adapter} [local, batch {a.batch}]", flush=True)
    model, tok = load(a.model, adapter_path=a.adapter); mx.random.seed(a.seed); sampler = make_sampler(a.temperature, a.top_p)
    per = max(1, a.batch // a.k); t0 = time.time(); stats = collections.Counter(); toks = 0; n_done = 0
    for b in range(0, len(todo), per):
        chunk = todo[b:b + per]
        prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': build_prompt(p['statement'])}], add_generation_prompt=True) for p in chunk]
        tg = time.time(); resp = batch_generate(model, tokenizer=tok, prompts=[pr for pr in prompts for _ in range(a.k)], max_tokens=a.max_tokens, sampler=sampler, verbose=False)
        gen_s = time.time() - tg; texts = resp.texts
        with open(a.out, 'a') as f:
            for j, p in enumerate(chunk):
                res = []
                for i in range(a.k):
                    t = texts[j * a.k + i]; ntok = len(tok.encode(t)); code = extract(t); g = grade(code, p, a.timeout, mem)
                    res.append(dict(id=p['id'], sample=i, source=p['source'], difficulty=p['difficulty'], **g, tokens=ntok, seconds=round(gen_s / len(texts), 2),
                                    truncated=ntok >= a.max_tokens - 2, code=code, text=t))
                    f.write(json.dumps(res[-1], ensure_ascii=False) + '\n')
                k_pass = sum(r['passed'] for r in res); stats[k_pass] += 1; toks += sum(r['tokens'] for r in res); n_done += 1
                el = time.time() - t0; eta = el / n_done * (len(todo) - n_done)
                print(f"  {n_done:>4}/{len(todo)} {p['id']:<12} {str(p['difficulty']):<16} pass {k_pass}/{a.k}  tok {[r['tokens'] for r in res]}  "
                      f"{el/60:5.1f} min, ETA {eta/60:4.0f} min  {res[0]['error'][:70] if not res[0]['passed'] else ''}", flush=True)
    wall = time.time() - t0
    print(f"done: {n_done} problems in {wall/60:.1f} min, tokens {toks} (~{toks/wall:.0f} tok/s total), distribution of passed samples out of {a.k}: {dict(sorted(stats.items()))} -> {a.out}")

if __name__ == '__main__': main()
