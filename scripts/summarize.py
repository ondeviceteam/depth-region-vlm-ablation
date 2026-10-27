#!/usr/bin/env python3
"""Score saved answers and write the result tables. No inference is run.

usage: python scripts/summarize.py RESULTS_DIR
RESULTS_DIR contains raw/*.jsonl written by run_eval.py.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counting_rca import parse_count, score_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'experiments/2026-10-28_7b_vqav2_gqa/protocol.json').read_text())
METHODS = ['Dense', 'ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']
COMPARISONS = [('Depth-RGB', 'ToSA'), ('Depth-only', 'ToSA'), ('Depth-RGB', 'ToMe'), ('Depth-only', 'ToMe'), ('ToSA', 'ToMe')]
NUMBER_WORDS = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}


def normalize(x):
    # VQAv2-style: lowercase, strip punctuation (a period only when no digit follows), drop articles, number words.
    x = x.lower().replace('\n', ' ').replace('\t', ' ').strip()
    x = re.sub(r'(?<!\d)[,!?;"()\[\]{}](?!\d)', ' ', x)
    x = re.sub(r'\.(?!\d)', ' ', x)
    return ' '.join(NUMBER_WORDS.get(t, t) for t in x.split() if t not in {'a', 'an', 'the'})


def vqa_accuracy(prediction, answers):
    p = normalize(prediction)
    g = [normalize(a) for a in answers]
    return sum(min(sum(p == x for j, x in enumerate(g) if j != i) / 3.0, 1.0) for i in range(len(g))) / len(g)


def score(dataset, row, q):
    ok = row['status'] == 'PASS'
    text = row['raw_response'] if ok else ''
    if dataset == 'vqav2_count':
        truth = int(q['multiple_choice_answer'])
        counting = score_count(parse_count(text) if ok else None, truth)
        return dict(vqa_accuracy=vqa_accuracy(text, q['answers']) if ok else 0.0,
                    rca=counting['rca_positive'], numeric_em=counting['numeric_em'])
    first = text.split('.')[0]
    return dict(exact_match=float(ok and normalize(first) == normalize(q['answer'])),
                tosa_rule=float(ok and first.strip() != '' and first.lower() in q['answer'].lower()))


def bootstrap(delta, seed=20261028, replicates=10000):
    index = np.random.default_rng(seed).integers(0, len(delta), (replicates, len(delta)))
    boot = delta[index].mean(1)
    return 100 * delta.mean(), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    out = Path(sys.argv[1]).resolve()
    rows = [json.loads(line) for p in sorted((out / 'raw').glob('*.jsonl')) for line in p.read_text().splitlines()]
    keeps = PROTOCOL['keep_settings']
    summary, paired, by_type, repeats = [], [], [], []
    for dataset, spec in PROTOCOL['datasets'].items():
        questions = {str(q['question_id']): q for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation']}
        cells = defaultdict(dict)
        for r in rows:
            if r['dataset'] != dataset:
                continue
            key = (r['method'], r['nominal_keep_ratio'], r['run'])
            qid = str(r['question_id'])
            assert qid not in cells[key], ('duplicate', key, qid)
            cells[key][qid] = dict(score(dataset, r, questions[qid]), status=r['status'], raw=r['raw_response'])
        if not cells:
            print('no rows yet for', dataset)
            continue
        expected = {(m, k, run) for m in METHODS for k in ([1.0] if m == 'Dense' else keeps)
                    for run in range(PROTOCOL['repeats'] if dataset == 'vqav2_count'
                                     and (m == 'Dense' or k == PROTOCOL['repeat_keep']) else 1)}
        assert set(cells) == expected, ('conditions', dataset, sorted(expected ^ set(cells)))
        for key, cell in cells.items():
            assert set(cell) == set(questions), ('incomplete', dataset, key, len(cell), len(questions))
        primary = 'vqa_accuracy' if dataset == 'vqav2_count' else 'exact_match'
        metrics = ['vqa_accuracy', 'rca', 'numeric_em'] if dataset == 'vqav2_count' else ['exact_match', 'tosa_rule']
        ids = sorted(questions)
        strength = PROTOCOL['merge_strength']
        for method in METHODS:
            for keep in ([1.0] if method == 'Dense' else keeps):
                cell = cells[(method, keep, 0)]
                row = dict(dataset=dataset, method=method, nominal_keep_ratio=keep,
                           merged_tokens=0 if keep == 1.0 else strength[str(keep)]['merged'],
                           remaining_tokens=729 if keep == 1.0 else strength[str(keep)]['remaining'],
                           completed=sum(v['status'] == 'PASS' for v in cell.values()), n=len(ids))
                for m in metrics:
                    values = [cell[i][m] for i in ids if cell[i][m] is not None]
                    row[m] = round(100 * float(np.mean(values)), 2)
                    row[f'{m}_n'] = len(values)
                summary.append(row)
                if dataset == 'gqa':
                    groups = defaultdict(list)
                    for i in ids:
                        groups[('semantic', questions[i]['semantic_type'])].append(cell[i]['exact_match'])
                        groups[('structural', questions[i]['structural_type'])].append(cell[i]['exact_match'])
                    for (kind, name), values in sorted(groups.items()):
                        by_type.append(dict(method=method, nominal_keep_ratio=keep, type_kind=kind, type=name,
                                            n=len(values), exact_match=round(100 * float(np.mean(values)), 2)))
        for keep in keeps:
            for a, b in COMPARISONS:
                for m in [primary] + (['rca'] if dataset == 'vqav2_count' else []):
                    use = [i for i in ids if cells[(a, keep, 0)][i][m] is not None]
                    delta = np.array([cells[(a, keep, 0)][i][m] - cells[(b, keep, 0)][i][m] for i in use])
                    d, lo, hi = bootstrap(delta)
                    paired.append(dict(dataset=dataset, nominal_keep_ratio=keep, metric=m, comparison=f'{a} - {b}',
                                       difference_pp=round(d, 2), ci95_low=round(lo, 2), ci95_high=round(hi, 2), n=len(use)))
        if dataset == 'vqav2_count':
            for method in METHODS:
                keep = 1.0 if method == 'Dense' else PROTOCOL['repeat_keep']
                runs = sorted(k[2] for k in cells if k[0] == method and k[1] == keep)
                scores = [100 * np.mean([cells[(method, keep, r)][i][primary] for i in ids]) for r in runs]
                agree = np.mean([len({cells[(method, keep, r)][i]['raw'] for r in runs}) == 1 for i in ids])
                repeats.append(dict(method=method, nominal_keep_ratio=keep, runs=len(runs),
                                    vqa_accuracy_runs=' / '.join(f'{s:.2f}' for s in scores),
                                    vqa_accuracy_range=round(max(scores) - min(scores), 2),
                                    identical_answer_rate=round(100 * agree, 2)))
    write(out / 'summary.csv', summary)
    write(out / 'paired_differences.csv', paired)
    write(out / 'gqa_by_type.csv', by_type)
    write(out / 'repeats.csv', repeats)
    print('rows', len(rows), 'summary', len(summary), 'paired', len(paired))


if __name__ == '__main__':
    main()
