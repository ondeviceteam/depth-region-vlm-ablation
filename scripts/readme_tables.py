#!/usr/bin/env python3
"""Print the README result tables (one per dataset and merge strength) from summary.csv.

usage: python scripts/readme_tables.py RESULTS_DIR
"""
import csv
import sys
from pathlib import Path

METHODS = ['ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']
COLUMNS = {'vqav2_count': [('vqa_accuracy', 'VQA accuracy (%)'), ('rca', 'RCA (%)')],
           'gqa': [('exact_match', 'Exact match (%)')]}
TITLE = {'vqav2_count': 'VQAv2 counting', 'gqa': 'GQA'}


def main():
    out = Path(sys.argv[1])
    rows = list(csv.DictReader((out / 'summary.csv').open()))
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
            best = {m: max(float(cell[x][m]) for x in METHODS) for m, _ in columns}
            n = base['n']
            print(f'| *Token merge 없음* | ' + ' | '.join(f"*{float(base[m]):.2f}*" for m, _ in columns) + f' | {base["completed"]}/{n} |')
            for method in METHODS:
                r = cell[method]
                values = [f'**{float(r[m]):.2f}**' if float(r[m]) == best[m] else f'{float(r[m]):.2f}' for m, _ in columns]
                print(f'| {method} | ' + ' | '.join(values) + f' | {r["completed"]}/{n} |')
            print()


if __name__ == '__main__':
    main()
