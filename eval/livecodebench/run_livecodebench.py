"""LiveCodeBench (code_generation_lite) pass@1 greedy on the data/subset.json subset, via an OpenAI-compatible endpoint.
Model code is run per test in a `python3 -I` subprocess in a temporary directory, with time and memory limits (RSS guard).
A problem counts as passed only if all public and private tests pass. Usage:
  python3 -I eval/livecodebench/run_livecodebench.py --model <id> [--limit N] [--workers 4] [--no-think] [--tag name] [--max-tokens 2048]
                                                     [--timeout 10] [--mem-mb 1024]"""
import json, os, sys, re, argparse, subprocess, tempfile, time, base64, zlib, pickle, io, hashlib, concurrent.futures as cf
from decimal import Decimal, InvalidOperation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.llm import chat
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f'{HERE}/data'

# --- prompt: faithful to lcb_runner/prompts/code_generation.py ("generic" template) ---
SYS = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct "
       "Python program that matches the specification and passes all tests.")
FMT_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
             "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")
def build_prompt(p):
    s = f"### Question:\n{p['question_content']}\n\n"
    if p['starter_code']: s += f"### Format: {FMT_STARTER}\n```python\n{p['starter_code']}\n```\n\n"
    else: s += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    return s + "### Answer: (use the provided format with backticks)\n\n"

def extract(text):
    m = re.findall(r'```(?:python|py|Python)?[ \t]*\n(.*?)```', text, re.S)
    if m: return m[-1]
    m = re.search(r'```(?:python|py|Python)?[ \t]*\n(.*)$', text, re.S)  # unclosed block
    return m.group(1) if m else text

# --- data: private tests are base64(zlib(pickle(JSON str))); unpickle only via an unpickler without globals (downloaded file = untrusted) ---
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name): raise pickle.UnpicklingError(f'forbidden global {module}.{name}')
def decode_tests(s):
    try: return json.loads(s)
    except Exception: pass
    obj = _SafeUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load()
    if not isinstance(obj, str): raise ValueError('unexpected pickle payload')
    return json.loads(obj)
def load_problems(limit=0):
    sub = json.load(open(f'{DATA}/subset.json'))
    rows = {}
    for l in open(f"{DATA}/{sub['source']['file']}"):
        r = json.loads(l); rows[r['question_id']] = r
    probs = []
    for meta in sub['problems'][:limit or None]:
        r = rows[meta['question_id']]
        tests = json.loads(r['public_test_cases']) + decode_tests(r['private_test_cases'])
        probs.append(dict(r, tests=tests, func_name=json.loads(r['metadata']).get('func_name')))
    return sub, probs

# --- execution: one subprocess per test, python3 -I, empty cwd, minimal env, time and RSS limits ---
IMPORTS = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\n"
           "from copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
           "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
           "import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json\n"
           "sys.setrecursionlimit(50000)\n")
FUNC_RUNNER = ("\n\nif __name__ == '__main__':\n    import json as _json, sys as _sys\n    _args = _json.load(open(_sys.argv[1]))\n"
               "    _res = Solution().{fn}(*_args)\n    if isinstance(_res, tuple): _res = list(_res)\n    print(_json.dumps(_res))\n")
ENV = {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'LANG': 'C.UTF-8'}
def _rss_bytes(pid):
    try: return int(subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True).stdout.strip() or 0) * 1024
    except Exception: return 0
def _soft_limits(mem):
    import resource
    for name in ('RLIMIT_AS', 'RLIMIT_DATA'):  # best-effort; macOS usually rejects it, the RSS guard actually enforces the limit
        try: resource.setrlimit(getattr(resource, name), (mem, mem))
        except Exception: pass
def run_proc(argv, cwd, stdin_path, timeout, mem):
    """-> (status, stdout, stderr); status: ok | timeout | memory | error"""
    with open(stdin_path, 'rb') if stdin_path else open(os.devnull, 'rb') as fin, tempfile.TemporaryFile() as fout, tempfile.TemporaryFile() as ferr:
        p = subprocess.Popen(argv, cwd=cwd, stdin=fin, stdout=fout, stderr=ferr, env=ENV, preexec_fn=lambda: _soft_limits(mem))
        t0 = time.time(); status = 'ok'
        while p.poll() is None:
            if time.time() - t0 > timeout: p.kill(); status = 'timeout'; break
            if _rss_bytes(p.pid) > mem: p.kill(); status = 'memory'; break
            time.sleep(0.01)
        p.wait()
        fout.seek(0); ferr.seek(0); out = fout.read(50_000_000).decode('utf-8', 'replace'); err = ferr.read(10_000).decode('utf-8', 'replace')
    if status == 'ok' and p.returncode != 0: status = 'error'
    return status, out, err

def _lines(s): s = s.strip(); return [l.strip() for l in s.split('\n')] if s else []
def _decimals(line):
    try: return [Decimal(x) for x in line.split()]
    except InvalidOperation: return None
def stdout_matches(got, exp):
    a, b = _lines(got), _lines(exp)
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if x == y: continue
        dx, dy = _decimals(x), _decimals(y)
        if dx is None or dy is None or dx != dy: return False
    return True

def grade(code, p, timeout, mem):
    """-> dict(passed, failed_test_count, tests_run, tests_total, error, stopped_early). Stops after a timeout/memory limit/SyntaxError (the remaining tests
    would end the same way, and 40 tests × 10 s is 7 min per problem)."""
    tests = p['tests']; failed = 0; run = 0; err = ''; stopped = False
    with tempfile.TemporaryDirectory() as d:
        if p['func_name']: prog = IMPORTS + '\n' + code + FUNC_RUNNER.format(fn=p['func_name'])
        else: prog = code
        prog_path = os.path.join(d, 'prog.py'); open(prog_path, 'w').write(prog)
        for i, t in enumerate(tests):
            inp = os.path.join(d, 'in.txt')
            if p['func_name']:
                args = [json.loads(l) for l in t['input'].split('\n') if l.strip() != '']
                json.dump(args, open(inp, 'w')); status, out, serr = run_proc([sys.executable, '-I', prog_path, inp], d, None, timeout, mem)
            else:
                open(inp, 'w').write(t['input'] if t['input'].endswith('\n') else t['input'] + '\n')
                status, out, serr = run_proc([sys.executable, '-I', prog_path], d, inp, timeout, mem)
            run += 1
            if status != 'ok':
                failed += 1; err = err or f'test {i}: {status} {serr.strip()[-200:]}'
                if status in ('timeout', 'memory') or 'SyntaxError' in serr or 'IndentationError' in serr: stopped = True; break
                continue
            if p['func_name']:
                try: ok = json.loads(out) == json.loads(t['output'])
                except Exception: ok = False
            else: ok = stdout_matches(out, t['output'])
            if not ok:
                failed += 1; err = err or f'test {i}: wrong answer, got {out.strip()[:80]!r} expected {t["output"].strip()[:80]!r}'
    return dict(passed=failed == 0 and run == len(tests), failed_test_count=failed, tests_run=run, tests_total=len(tests), error=err, stopped_early=stopped)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--no-think', action='store_true'); ap.add_argument('--tag', default='')
    ap.add_argument('--max-tokens', type=int, default=2048); ap.add_argument('--http-timeout', type=float, default=600.0, help='timeout of a single HTTP request (s); thinking-mode models with a 32k budget need ~3600'); ap.add_argument('--timeout', type=float, default=10.0); ap.add_argument('--mem-mb', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=0.0, help='0 = greedy (default, reproducible). Qwen recommends 0.7 in non-thinking mode / 0.6 in thinking mode')
    ap.add_argument('--top-p', type=float, default=1.0); ap.add_argument('--top-k', type=int, default=0, help='0 = disabled; Qwen recommends 20')
    ap.add_argument('--retries', type=int, default=3, help='how many times to retry a request after an API/HTTP error (a server under memory pressure drops connections)')
    ap.add_argument('--retry-from', default='', help='path to an existing results/*.json: re-run only the tasks that ended with an API error and overwrite that file')
    a = ap.parse_args(); mem = a.mem_mb * 1024 * 1024
    sub, probs = load_problems(a.limit)
    prev = None
    if a.retry_from:
        prev = json.load(open(a.retry_from)); redo = {p['id'] for p in prev['per_problem'] if p['error'].startswith('api')}
        probs = [p for p in probs if p['question_id'] in redo]; print(f"re-running {len(probs)} tasks with API errors from {a.retry_from}", flush=True)
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_think else {}
    if a.temperature > 0:  # with greedy decoding top_p/top_k change nothing, so we do not send them
        extra['top_p'] = a.top_p
        if a.top_k > 0: extra['top_k'] = a.top_k
    sampling = dict(temperature=a.temperature, top_p=a.top_p if a.temperature > 0 else None, top_k=a.top_k if a.temperature > 0 else None)
    def run_one(p):
        base = dict(id=p['question_id'], platform=p['platform'], difficulty=p['difficulty'], contest_date=p['contest_date'][:10])
        for attempt in range(a.retries + 1):
            try: r = chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': build_prompt(p)}], a.model, max_tokens=a.max_tokens, temperature=a.temperature, extra=extra or None, timeout=a.http_timeout); break
            except Exception as e:
                err = e; print(f"  api   {p['question_id']:<12} attempt {attempt + 1}/{a.retries + 1}: {str(e)[:80]}", flush=True)
                if attempt < a.retries: time.sleep(30 * (attempt + 1))
        else:
            return dict(base, passed=False, failed_test_count=0, tests_run=0, tests_total=len(p['tests']), error=f'api: {err}', seconds=0, tokens=0, truncated=False, code='')
        code = extract(r['text']); g = grade(code, p, a.timeout, mem)
        print(f"  {'PASS' if g['passed'] else 'fail'} {p['question_id']:<12} {p['difficulty']:<6} {g['tests_run']:>3}/{g['tests_total']:<3} tok {r['usage'].get('completion_tokens', 0):>5}  {g['error'][:90]}", flush=True)
        return dict(base, **g, seconds=r['seconds'], tokens=r['usage'].get('completion_tokens', 0), truncated=r['usage'].get('completion_tokens', 0) >= a.max_tokens, code=code)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex: results = list(ex.map(run_one, probs))
    if prev is not None:  # merge the re-run tasks into the old result and overwrite the file
        new = {r['id']: r for r in results}; results = [new.get(p['id'], p) for p in prev['per_problem']]
        t0 -= prev['wall_seconds']; a.tag = os.path.splitext(os.path.basename(a.retry_from))[0]
    n = len(results); k = sum(r['passed'] for r in results); gen_s = sum(r['seconds'] for r in results); toks = sum(r['tokens'] for r in results)
    by_diff = {d: f"{sum(r['passed'] for r in results if r['difficulty'] == d)}/{sum(r['difficulty'] == d for r in results)}" for d in ('easy', 'medium', 'hard')}
    tag = a.tag or re.sub(r'[^A-Za-z0-9._-]', '_', a.model)
    os.makedirs(f'{HERE}/results', exist_ok=True)
    json.dump(dict(model=a.model, no_think=a.no_think, max_tokens=a.max_tokens, sampling=sampling, timeout_s=a.timeout, mem_mb=a.mem_mb,
                   dataset=dict(sub['source'], subset_sha256=hashlib.sha256(open(f'{DATA}/subset.json', 'rb').read()).hexdigest(), window=sub['window']),
                   n=n, passed=k, pass_at_1=round(k / n, 4), by_difficulty=by_diff, wall_seconds=round(time.time() - t0, 1), completion_tokens=toks,
                   tok_per_s_per_stream=round(toks / gen_s, 1) if gen_s else None, per_problem=results),
              open(f'{HERE}/results/{tag}.json', 'w'), indent=1)
    print(f"{a.model}: pass@1 = {k}/{n} = {k/n:.3f}  {by_diff}  wall {time.time()-t0:.0f}s  tokens {toks}  ~{toks/gen_s if gen_s else 0:.0f} tok/s per stream")
if __name__ == '__main__': main()
