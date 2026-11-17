"""Collects several sampled LCB runs (pass@1 averaged over samples) and compares two models pairwise on problems.

usage: python3 summarize_samples.py <prefix_A> <prefix_B>
  e.g. summarize_samples.py flash-oq4e-norev-r6s05-nothink-8192-t07 flash-oq4e-nothink-8192-t07
Reads results/<prefix>-s*.json. For each problem computes the fraction of passed samples; sign test on problems where the fractions differ."""
import glob, json, os, statistics as st, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
def load(prefix):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f'{HERE}/results/{prefix}-s*.json'))]
    if not runs: sys.exit(f'no results for {prefix}')
    per = {}
    for r in runs:
        for p in r['per_problem']: per.setdefault(p['id'], []).append(p)
    return runs, per
def describe(name, runs, per):
    passed = [r['passed'] for r in runs]
    toks = [p['tokens'] for ps in per.values() for p in ps]; trunc = sum(p.get('truncated', False) for ps in per.values() for p in ps)
    hard = [sum(p['passed'] for p in ps) / len(ps) for ps in per.values() if ps[0]['difficulty'] == 'hard']
    print(f"{name}: {len(runs)} samples, pass@1 = {st.mean(passed):.1f} ± {st.pstdev(passed):.1f}  (samples: {passed})  "
          f"hard {sum(hard):.1f}/{len(hard)}  median tok {int(st.median(toks))}  truncated {trunc}/{len(toks)}  "
          f"time {sum(r['wall_seconds'] for r in runs)/60/len(runs):.0f} min/sample")
def sign_test(wins, losses):
    n = wins + losses
    if n == 0: return 1.0
    k = min(wins, losses); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def main():
    a, b = sys.argv[1:3]
    ra, pa = load(a); rb, pb = load(b)
    describe('A ' + a, ra, pa); describe('B ' + b, rb, pb)
    wins = losses = 0; ids = sorted(set(pa) & set(pb))
    for i in ids:
        fa = sum(p['passed'] for p in pa[i]) / len(pa[i]); fb = sum(p['passed'] for p in pb[i]) / len(pb[i])
        if fa > fb: wins += 1
        elif fb > fa: losses += 1
    print(f"paired on {len(ids)} problems: A better on {wins}, B better on {losses}, tie {len(ids)-wins-losses}; sign test p = {sign_test(wins, losses):.3f}")
    stable_a = sum(all(p['passed'] for p in pa[i]) for i in ids); stable_b = sum(all(p['passed'] for p in pb[i]) for i in ids)
    print(f"problems passed in all samples: A {stable_a}, B {stable_b}")
if __name__ == '__main__': main()
