#!/usr/bin/env python3
"""Score the relaxed Depth methods and compare them with the original answers of the same model. No inference is run.

usage: python scripts/summarize_relaxed.py RESULTS_DIR
RESULTS_DIR (results/2026-10-28_depth_relaxed) contains raw/*.jsonl written by run_relaxed_eval.py.
The original experiments of each model must already be summarized (results/2026-10-28_<model>_vqav2_gqa/raw).

Outputs:
  summary.csv             score per model, dataset, relaxed method and merge strength (same columns as summary.py)
  paired_differences.csv  pre-specified comparisons with the original ToSA and ToMe (prespecified=True),
                          plus relaxed - original Depth method (prespecified=False)
  consistency.csv         questions whose threshold stays 0.15: rate of answers identical to the original Depth method
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ROOT, bootstrap, load_protocol, score, write  # noqa: E402

ORIGINAL = {'7B': 'results/2026-10-28_7b_vqav2_gqa', '0.5B': 'results/2026-10-28_0.5b_vqav2_gqa'}
BASE = {'Depth-only-relaxed': 'Depth-only', 'Depth-RGB-relaxed': 'Depth-RGB'}
COMPARISONS = [('Depth-RGB-relaxed', 'ToSA', True), ('Depth-only-relaxed', 'ToSA', True),
               ('Depth-RGB-relaxed', 'ToMe', True), ('Depth-only-relaxed', 'ToMe', True),
               ('Depth-RGB-relaxed', 'Depth-RGB', False), ('Depth-only-relaxed', 'Depth-only', False)]


def load(paths, dataset, methods, keeps):
    cells = defaultdict(dict)
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['dataset'] != dataset or row['method'] not in methods or row['run'] != 0 \
                    or row['nominal_keep_ratio'] not in keeps:
                continue
            key = (row['method'], row['nominal_keep_ratio'])
            qid = str(row['question_id'])
            assert qid not in cells[key], ('duplicate', key, qid)
            cells[key][qid] = row
    return cells


def main():
    out = Path(sys.argv[1]).resolve()
    protocol = load_protocol(out)
    keeps = protocol['keep_settings']
    strength = protocol['merge_strength']
    thresholds = json.loads((ROOT / protocol['thresholds']['file']).read_text())['images']
    summary, paired, consistency = [], [], []
    for model in protocol['models']:
        relaxed_paths = sorted((out / 'raw').glob(f'{model}_*.jsonl'))
        original_paths = sorted((ROOT / ORIGINAL[model] / 'raw').glob('*.jsonl'))
        for dataset, spec in protocol['datasets'].items():
            questions = {str(q['question_id']): q for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation']}
            ids = sorted(questions)
            relaxed = load(relaxed_paths, dataset, set(BASE), set(keeps))
            if not relaxed:
                print('no rows yet for', model, dataset)
                continue
            original = load(original_paths, dataset, {'ToMe', 'ToSA', 'Depth-only', 'Depth-RGB'}, set(keeps))
            cells = {**relaxed, **original}
            for key in [(m, k) for m in list(BASE) + ['ToMe', 'ToSA', 'Depth-only', 'Depth-RGB'] for k in keeps]:
                assert set(cells.get(key, {})) == set(questions), ('incomplete', model, dataset, key, len(cells.get(key, {})))
            scored = {key: {i: score(dataset, cell[i], questions[i]) for i in ids} for key, cell in cells.items()}
            primary = 'vqa_accuracy' if dataset == 'vqav2_count' else 'exact_match'
            metrics = ['vqa_accuracy', 'rca', 'numeric_em'] if dataset == 'vqav2_count' else ['exact_match', 'tosa_rule']
            for method in BASE:
                for keep in keeps:
                    cell, s = cells[(method, keep)], scored[(method, keep)]
                    raised = [i for i in ids if cell[i]['boundary_threshold'] != 0.15]
                    row = dict(model=model, dataset=dataset, method=method, nominal_keep_ratio=keep,
                               merged_tokens=strength[str(keep)]['merged'], remaining_tokens=strength[str(keep)]['remaining'],
                               completed=sum(cell[i]['status'] == 'PASS' for i in ids), n=len(ids),
                               raised_threshold_questions=len(raised))
                    for m in metrics:
                        values = [s[i][m] for i in ids if s[i][m] is not None]
                        row[m] = round(100 * float(np.mean(values)), 2)
                        row[f'{m}_n'] = len(values)
                    summary.append(row)
                    # The threshold recorded in each answer must be the frozen one for its image.
                    target = str(strength[str(keep)]['remaining'])
                    for i in ids:
                        stem = Path(questions[i]['local_image_path']).stem
                        assert cell[i]['boundary_threshold'] == thresholds[dataset][stem][target]['threshold'], (model, i)
                    same = [i for i in ids if cell[i]['boundary_threshold'] == 0.15]
                    base = cells[(BASE[method], keep)]
                    consistency.append(dict(model=model, dataset=dataset, method=method, nominal_keep_ratio=keep,
                                            threshold_015_questions=len(same),
                                            identical_answers=sum(cell[i]['raw_response'] == base[i]['raw_response'] for i in same),
                                            identical_answer_rate=round(100 * np.mean(
                                                [cell[i]['raw_response'] == base[i]['raw_response'] for i in same]), 2) if same else None))
            for keep in keeps:
                for a, b, prespecified in COMPARISONS:
                    for m in [primary] + (['rca'] if dataset == 'vqav2_count' else []):
                        use = [i for i in ids if scored[(a, keep)][i][m] is not None and scored[(b, keep)][i][m] is not None]
                        delta = np.array([scored[(a, keep)][i][m] - scored[(b, keep)][i][m] for i in use])
                        d, lo, hi = bootstrap(delta)
                        paired.append(dict(model=model, dataset=dataset, nominal_keep_ratio=keep, metric=m,
                                           comparison=f'{a} - {b}', prespecified=prespecified and m == primary,
                                           difference_pp=round(d, 2), ci95_low=round(lo, 2), ci95_high=round(hi, 2), n=len(use)))
    write(out / 'summary.csv', summary)
    write(out / 'paired_differences.csv', paired)
    write(out / 'consistency.csv', consistency)
    print('summary', len(summary), 'paired', len(paired), 'consistency', len(consistency))


if __name__ == '__main__':
    main()
