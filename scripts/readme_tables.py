#!/usr/bin/env python3
"""Print the README result tables (one per dataset and merge strength) from summary.csv.

usage: python scripts/readme_tables.py RESULTS_DIR [RELAXED_DIR]
With RELAXED_DIR (results/2026-10-28_depth_relaxed), the relaxed Depth methods of the same model are added as extra rows.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']
LABELS = {'Depth-only-relaxed': 'Depth-only (경계 완화)', 'Depth-RGB-relaxed': 'Depth-RGB (경계 완화)'}
COLUMNS = {'vqav2_count': [('vqa_accuracy', 'VQA accuracy (%)'), ('rca', 'RCA (%)')],
           'gqa': [('exact_match', 'Exact match (%)')]}
TITLE = {'vqav2_count': 'VQAv2 counting', 'gqa': 'GQA'}


def relaxed_rows(results, relaxed):
    # Keep the relaxed rows of the model that the original experiment used.
    name = json.loads((ROOT / 'experiments' / results.name / 'protocol.json').read_text())['model']
    models = json.loads((ROOT / 'experiments' / relaxed.name / 'protocol.json').read_text())['models']
    key = next(k for k, v in models.items() if v['name'] == name)
    return [r for r in csv.DictReader((relaxed / 'summary.csv').open()) if r['model'] == key]


def main():
    out = Path(sys.argv[1]).resolve()
    rows = list(csv.DictReader((out / 'summary.csv').open()))
    methods = list(METHODS)
    if len(sys.argv) > 2:
        rows += relaxed_rows(out, Path(sys.argv[2]).resolve())
        methods += list(LABELS)
    for dataset, columns in COLUMNS.items():
        data = [r for r in rows if r['dataset'] == dataset]
        if not data:
            continue
        base = next(r for r in data if r['method'] == 'Dense')
        keeps = sorted({r['nominal_keep_ratio'] for r in data if r['method'] != 'Dense'}, key=float, reverse=True)
        for keep in keeps:
            cell = {r['method']: r for r in data if r['nominal_keep_ratio'] == keep}
            merged, remaining = int(cell['ToMe']['merged_tokens']), int(cell['ToMe']['remaining_tokens'])
            print(f'### {TITLE[dataset]} · {merged}개 병합 ({100 * merged / 729:.1f}% 병합, 729 → {remaining}개 남음)\n')
            print('| 방법 | ' + ' | '.join(c[1] for c in columns) + ' | 실행 성공 |')
            print('|---|' + '---:|' * (len(columns) + 1))
            best = {m: max(float(cell[x][m]) for x in methods) for m, _ in columns}
            n = base['n']
            print(f'| *Token merge 없음* | ' + ' | '.join(f"*{float(base[m]):.2f}*" for m, _ in columns) + f' | {base["completed"]}/{n} |')
            for method in methods:
                r = cell[method]
                values = [f'**{float(r[m]):.2f}**' if float(r[m]) == best[m] else f'{float(r[m]):.2f}' for m, _ in columns]
                print(f'| {LABELS.get(method, method)} | ' + ' | '.join(values) + f' | {r["completed"]}/{n} |')
            print()


if __name__ == '__main__':
    main()
