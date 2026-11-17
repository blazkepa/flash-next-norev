"""Builds data/subset.json: a fixed window of LiveCodeBench problems for measuring regressions. Run once; the result is versioned.
  python3 -I eval/livecodebench/make_subset.py
Rule: problems published after the base model's data cutoff date, at most 150; if there are fewer than 60, the newest 100 (contamination risk)."""
import json, hashlib, os, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f'{HERE}/data'
SRC = dict(dataset='livecodebench/code_generation_lite', revision='0fe84c3912ea0c4d4a78037083943e8f0c4dd505', version_tag='release_v6',
           file='test6.jsonl', note='test6.jsonl = v6 increment (problems 2025-01-04..2025-04-06, 175 total); full release_v6 = test.jsonl..test6.jsonl, 1055 problems')
# Cutoff date: the Qwen/Qwen3.5-4B model card does not state it explicitly. Indirectly: the card reports results on HMMT Nov 25 and LiveCodeBench v6,
# and the model was released around 2026-02-24. We conservatively assume the training data reaches at least 2025-11-30.
CUTOFF = '2025-11-30'
rows = [json.loads(l) for l in open(f'{DATA}/{SRC["file"]}')]
sha = hashlib.sha256(open(f'{DATA}/{SRC["file"]}', 'rb').read()).hexdigest()
after = [r for r in rows if r['contest_date'][:10] > CUTOFF]
if len(after) >= 60:
    chosen = sorted(after, key=lambda r: r['contest_date'])[:150]; rule = f'problems after {CUTOFF}, at most 150'; contaminated = False
else:
    chosen = sorted(rows, key=lambda r: r['contest_date'])[-100:]; rule = f'only {len(after)} problems after {CUTOFF} (< 60), so the newest 100 from {SRC["file"]}'; contaminated = True
chosen.sort(key=lambda r: (r['contest_date'], r['platform'], r['question_id']))
diff = {d: sum(r['difficulty'] == d for r in chosen) for d in ('easy', 'medium', 'hard')}
plat = {p: sum(r['platform'] == p for r in chosen) for p in sorted({r['platform'] for r in chosen})}
out = dict(created=datetime.date.today().isoformat(), source=dict(SRC, sha256=sha), base_model='Qwen/Qwen3.5-4B', assumed_cutoff=CUTOFF,
           rule=rule, contamination_risk=contaminated,
           rationale=('All problems in the newest available LiveCodeBench (release_v6, up to 2025-04-06) are older than the presumed data cutoff of '
                      'Qwen3.5 (>= 2025-11, the model card reports HMMT Nov 25). The set is therefore not contamination-free; it serves as a second regression guard, '
                      'harder than HumanEval (contest problems with hidden tests, stdin/stdout and function-style), not as an absolute measure. '
                      'Selected: the newest 100, sorted by date ascending. Compare only within the same subset.json.'),
           window=dict(from_date=chosen[0]['contest_date'][:10], to_date=chosen[-1]['contest_date'][:10]), n=len(chosen), difficulty=diff, platform=plat,
           problems=[dict(question_id=r['question_id'], platform=r['platform'], contest_id=r['contest_id'], contest_date=r['contest_date'][:10], difficulty=r['difficulty']) for r in chosen])
json.dump(out, open(f'{DATA}/subset.json', 'w'), indent=1, ensure_ascii=False)
print(f"n={out['n']} window {out['window']} difficulty {diff} platforms {plat} contaminated={contaminated}\nrule: {rule}\nsha256 {sha}")
