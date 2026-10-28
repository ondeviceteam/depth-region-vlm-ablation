#!/usr/bin/env python3
"""Score the ToSA merge amount x quantization answers and write the result tables. No inference is run.

usage: python scripts/summarize_quant.py RESULTS_DIR
RESULTS_DIR (results/<experiment>) contains raw/*.jsonl written by run_quant_eval.py.

Outputs:
  scores.csv        score per model, precision, dataset and merge amount
  quant_loss.csv    Q(p, r) = score(p, r) - score(FP, r) with paired bootstrap intervals
  interaction.csv   I(p, r) = Q(p, r) - Q(p, 0) for the pre-specified r (primary test)
  repeats.csv       run-to-run variation at the repeated merge amount
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ROOT, bootstrap, load_protocol, score, write  # noqa: E402

PRIMARY = {'vqav2_count': 'vqa_accuracy', 'gqa': 'exact_match'}
TESTED_R = [13, 25]


def main():
    out = Path(sys.argv[1]).resolve()
    protocol = load_protocol(out)
    precisions = list(protocol['precisions'])
    r_values = protocol['r_values']
    cells = defaultdict(dict)
    for path in sorted((out / 'raw').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['model'], row['precision'], row['dataset'], row['r_per_layer'], row['run'])
            qid = str(row['question_id'])
            assert qid not in cells[key], ('duplicate', key, qid)
            cells[key][qid] = row
    scores, losses, interactions, repeats = [], [], [], []
    for dataset, spec in protocol['datasets'].items():
        questions = {str(q['question_id']): q for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation']}
        ids = sorted(questions)
        metrics = ['vqa_accuracy', 'rca'] if dataset == 'vqav2_count' else ['exact_match']
        for model in protocol['models']:
            expected = {(model, p, dataset, r, run) for p in precisions for r in r_values
                        for run in range(protocol['repeats'] if dataset == 'vqav2_count' and r == protocol['repeat_r'] else 1)}
            present = {k for k in cells if k[0] == model and k[2] == dataset}
            if not present:
                print('no rows yet for', model, dataset)
                continue
            assert present == expected, ('conditions', model, dataset, sorted(present ^ expected))
            per = {}
            for key in expected:
                assert set(cells[key]) == set(questions), ('incomplete', key, len(cells[key]))
                per[key] = {i: score(dataset, cells[key][i], questions[i]) for i in ids}
            vec = lambda p, r, m: np.array([np.nan if per[(model, p, dataset, r, 0)][i][m] is None
                                            else per[(model, p, dataset, r, 0)][i][m] for i in ids], dtype=float)
            for p in precisions:
                for r in r_values:
                    row = dict(model=model, precision=p, dataset=dataset, r_per_layer=r, merged_tokens=26 * r,
                               remaining_tokens=729 - 26 * r, n=len(ids))
                    for m in metrics:
                        v = vec(p, r, m)
                        row[m] = round(100 * float(np.nanmean(v)), 2)
                    scores.append(row)
            for m in metrics:
                for p in precisions[1:]:
                    q = {r: vec(p, r, m) - vec('FP', r, m) for r in r_values}
                    for r in r_values:
                        d = q[r][~np.isnan(q[r])]
                        mean, lo, hi = bootstrap(d)
                        losses.append(dict(model=model, precision=p, dataset=dataset, metric=m, r_per_layer=r,
                                           merged_tokens=26 * r, quant_loss_pp=round(mean, 2),
                                           ci95_low=round(lo, 2), ci95_high=round(hi, 2), n=len(d)))
                    for r in TESTED_R:
                        d = q[r] - q[0]
                        d = d[~np.isnan(d)]
                        mean, lo, hi = bootstrap(d)
                        verdict = 'larger loss with merging' if hi < 0 else 'smaller loss with merging' if lo > 0 else 'uncertain'
                        interactions.append(dict(model=model, precision=p, dataset=dataset, metric=m, r_per_layer=r,
                                                 merged_tokens=26 * r, interaction_pp=round(mean, 2),
                                                 ci95_low=round(lo, 2), ci95_high=round(hi, 2), n=len(d),
                                                 primary=m == PRIMARY[dataset], reading=verdict))
            if dataset == 'vqav2_count':
                r = protocol['repeat_r']
                for p in precisions:
                    runs = range(protocol['repeats'])
                    s = [100 * np.mean([per[(model, p, dataset, r, k)][i]['vqa_accuracy'] for i in ids]) for k in runs]
                    same = np.mean([len({cells[(model, p, dataset, r, k)][i]['raw_response'] for k in runs}) == 1 for i in ids])
                    repeats.append(dict(model=model, precision=p, r_per_layer=r, runs=len(s),
                                        vqa_accuracy_runs=' / '.join(f'{x:.2f}' for x in s),
                                        vqa_accuracy_range=round(max(s) - min(s), 2), identical_answer_rate=round(100 * same, 2)))
    write(out / 'scores.csv', scores)
    write(out / 'quant_loss.csv', losses)
    write(out / 'interaction.csv', interactions)
    write(out / 'repeats.csv', repeats)
    print('scores', len(scores), 'quant_loss', len(losses), 'interaction', len(interactions))


if __name__ == '__main__':
    main()
