#!/usr/bin/env python3
"""In-process A/B test (mlx_vlm): greedy non-thinking generation on selected open-r1 problems, graded with tests (grade from LCB).
train/.venv/bin/python -I tools/flash_ab_generate.py --model <view> --ids a,b,c --out file.jsonl [--max-tokens 2500]"""
import json, os, sys, argparse, time, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{ROOT}/eval/livecodebench'); sys.path.insert(0, f'{ROOT}/eval')
import mlx.core as mx
from d1_sample import load_problems, build_prompt, SYS
from run_livecodebench import grade, extract
from flash_find_direction import SUFFIX
REV = re.compile(r"\b(Wait|Actually|Hmm|Let me reconsider|Let me re-?check|But wait|Alternatively|Let me re-?think)\b")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--ids', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=2500); ap.add_argument('--nprob', type=int, default=1000, help='how many problems to load from open-r1 (held-out 1000-1099 need 1100)'); ap.add_argument('--timeout', type=float, default=10.0); ap.add_argument('--mem-mb', type=int, default=1024)
    a = ap.parse_args(); ids = a.ids.split(',')
    probs, _ = load_problems(a.nprob, 5, 50, 7); P = {p['id']: p for p in probs}
    mx.set_cache_limit(2 * 1024**3)
    from mlx_vlm import load, generate
    model, proc = load(a.model, lazy=False, strict=False); tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    print(f"{a.model}: loaded, {mx.get_active_memory()/1e9:.1f} GB", flush=True); t0 = time.time()
    with open(a.out, 'w') as f:
        for pid in ids:
            p = P[pid]; msgs = [{'role': 'system', 'content': SYS + '\n' + SUFFIX}, {'role': 'user', 'content': build_prompt(p['statement'])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            t = time.time(); out = generate(model, proc, prompt, max_tokens=a.max_tokens, verbose=False, temperature=0.0); txt = out.text if hasattr(out, 'text') else out
            ntok = len(tok(txt, add_special_tokens=False)['input_ids']); trunc = ntok >= a.max_tokens - 2; code = extract(txt); g = grade(code, p, a.timeout, a.mem_mb * 1024 * 1024)
            row = dict(id=pid, difficulty=p.get('difficulty'), passed=g['passed'], tests_run=g['tests_run'], tests_total=g['tests_total'], error=g.get('error', ''), tokens=ntok, truncated=trunc,
                       revisions=len(REV.findall(txt)), has_code=('`' * 3) in txt, seconds=round(time.time() - t, 1), text=txt); f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            print(f"  {pid:12s} {str(p.get('difficulty'))[:12]:12s} {'PASS' if g['passed'] else 'fail':4s} tok {ntok:5d} {'trunc.' if trunc else '      '} revisions {row['revisions']:2d} code {int(row['has_code'])} {row['seconds']:5.0f}s  {str(g.get('error',''))[:50]}", flush=True)
            mx.clear_cache()
    print(f"done {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__': main()
