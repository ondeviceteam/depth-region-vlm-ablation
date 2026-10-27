#!/usr/bin/env python3
"""Post-hoc analysis (not in the protocol): score every method only on the questions where the
depth methods could run (no STRUCTURAL_INFEASIBLE), so infeasibility and merge quality are separated.

usage: python scripts/feasible_subset.py RESULTS_DIR
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import PROTOCOL, ROOT, score  # noqa: E402

METHODS = ['Dense', 'ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']


def main():
    out = Path(sys.argv[1]).resolve()
    rows = [json.loads(line) for p in sorted((out / 'raw').glob('*.jsonl')) for line in p.read_text().splitlines()]
    cells = defaultdict(dict)
    for r in rows:
        if r['run'] == 0:
            cells[(r['dataset'], r['method'], r['nominal_keep_ratio'])][str(r['question_id'])] = r
    result = []
    for dataset, spec in PROTOCOL['datasets'].items():
        questions = {str(q['question_id']): q for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation']}
        metric = 'vqa_accuracy' if dataset == 'vqav2_count' else 'exact_match'
        for keep in PROTOCOL['keep_settings']:
            # Depth-only and Depth-RGB share the same constraints, so their feasible sets are identical.
            feasible = sorted(q for q, r in cells[(dataset, 'Depth-only', keep)].items() if r['status'] == 'PASS')
            assert feasible == sorted(q for q, r in cells[(dataset, 'Depth-RGB', keep)].items() if r['status'] == 'PASS')
            row = dict(dataset=dataset, nominal_keep_ratio=keep, metric=metric, feasible=len(feasible), n=len(questions))
            for method in METHODS:
                cell = cells[(dataset, method, 1.0 if method == 'Dense' else keep)]
                row[method] = round(100 * sum(score(dataset, cell[q], questions[q])[metric] for q in feasible) / len(feasible), 2)
            result.append(row)
    with (out / 'feasible_subset.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    for row in result:
        print(row)


if __name__ == '__main__':
    main()
