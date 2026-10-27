#!/usr/bin/env python3
"""Build the GQA testdev-balanced manifest and write its 398 images.

Questions come from the official questions1.2.zip
(testdev_balanced_questions.json). Images come from the lmms-lab GQA
testdev_balanced_images parquet, written unchanged as <imageId>.jpg.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--questions-zip', type=Path, required=True)
    ap.add_argument('--images-parquet', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with zipfile.ZipFile(args.questions_zip) as z:
        name = next(n for n in z.namelist() if n.endswith('testdev_balanced_questions.json'))
        questions = json.loads(z.read(name))
    args.images.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.images_parquet).to_pylist()
    image_sha = {}
    for row in table:
        path = args.images / f"{row['id']}.jpg"
        if not path.exists():
            path.write_bytes(row['image']['bytes'])
        image_sha[row['id']] = sha(path)
    rows = []
    for qid in sorted(questions, key=int):
        q = questions[qid]
        assert q['imageId'] in image_sha, q['imageId']
        rows.append(dict(question_id=qid, image_id=q['imageId'], question=q['question'], answer=q['answer'],
                         structural_type=q['types']['structural'], semantic_type=q['types']['semantic'],
                         local_image_path=f"{q['imageId']}.jpg", image_sha256=image_sha[q['imageId']]))
    manifest = dict(dataset='GQA testdev-balanced', questions=len(rows), images=len(image_sha),
                    sources={'questions_zip': 'https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip',
                             'questions_zip_sha256': sha(args.questions_zip), 'questions_file': name,
                             'images_parquet': 'https://huggingface.co/datasets/lmms-lab/GQA (testdev_balanced_images)',
                             'images_parquet_sha256': sha(args.images_parquet)},
                    preflight_note='Preflight uses the first 5 questions; it checks code paths only and is not scored separately.',
                    evaluation=rows, preflight=rows[:5])
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print('written', args.output, len(rows), 'questions', len(image_sha), 'images')


if __name__ == '__main__':
    main()
