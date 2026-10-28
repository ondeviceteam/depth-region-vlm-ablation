#!/usr/bin/env python3
"""Draw a fixed random subset of the GQA testdev-balanced manifest.

Questions are ranked by SHA256(f"{seed}:{question_id}") and the first COUNT are
kept, sorted by question_id. Preflight questions are copied unchanged.

usage: python scripts/make_gqa_subset.py --count 2000 --seed 20261028
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'datasets/gqa_testdev_balanced.json')
    ap.add_argument('--output', type=Path, default=ROOT / 'datasets/gqa_testdev_balanced_2000.json')
    ap.add_argument('--count', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261028)
    args = ap.parse_args()
    source = json.loads(args.source.read_text())
    rank = lambda q: hashlib.sha256(f"{args.seed}:{q['question_id']}".encode()).hexdigest()
    rows = sorted(sorted(source['evaluation'], key=rank)[:args.count], key=lambda q: int(q['question_id']))
    manifest = dict(dataset=f"GQA testdev-balanced, {args.count}-question random subset",
                    selection=f"First {args.count} questions by SHA256('{args.seed}:<question_id>') from {args.source.name}",
                    source_manifest=args.source.name,
                    source_manifest_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    questions=len(rows), images=len({q['image_id'] for q in rows}),
                    preflight_note=source['preflight_note'], evaluation=rows, preflight=source['preflight'])
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print('written', args.output, len(rows), 'questions', manifest['images'], 'images')


if __name__ == '__main__':
    main()
