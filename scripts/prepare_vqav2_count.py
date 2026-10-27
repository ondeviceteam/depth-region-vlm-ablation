#!/usr/bin/env python3
"""Select VQAv2 val counting questions and download their COCO images.

Eligible: answer_type == number, question starts with "how many", the
representative answer is a nonnegative integer, and the image is not in the
excluded list. Images are ranked by SHA256(f"{seed}:{image_id}"); each image
contributes its lowest eligible question_id. The first `count` images form the
evaluation set and the next `preflight` images the preflight set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

COCO_URL = 'http://images.cocodataset.org/val2014/{}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_zip_json(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.json')]
        assert len(names) == 1, names
        return json.loads(z.read(names[0]))


def download(url, path):
    if path.exists():
        return
    temporary = path.with_name(path.name + '.part')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
                out.write(response.read())
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--questions-zip', type=Path, required=True)
    ap.add_argument('--annotations-zip', type=Path, required=True)
    ap.add_argument('--excluded', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--preflight', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20261028)
    args = ap.parse_args()
    questions = {int(q['question_id']): q for q in load_zip_json(args.questions_zip)['questions']}
    annotations = load_zip_json(args.annotations_zip)['annotations']
    excluded = set(json.loads(args.excluded.read_text())['image_ids'])
    eligible, skipped_non_integer = {}, 0
    for a in annotations:
        qid, image = int(a['question_id']), int(a['image_id'])
        if image in excluded or a['answer_type'] != 'number':
            continue
        if not questions[qid]['question'].lower().startswith('how many'):
            continue
        if not a['multiple_choice_answer'].isdigit():
            skipped_non_integer += 1
            continue
        if image not in eligible or qid < eligible[image][0]:
            eligible[image] = (qid, a)
    ranked = sorted(eligible, key=lambda i: (hashlib.sha256(f'{args.seed}:{i}'.encode()).hexdigest(), i))
    need = args.count + args.preflight
    assert len(ranked) >= need, (len(ranked), need)
    args.images.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, image in enumerate(ranked[:need]):
        qid, a = eligible[image]
        name = f'COCO_val2014_{image:012d}.jpg'
        download(COCO_URL.format(name), args.images / name)
        rows.append(dict(question_id=qid, image_id=image, question=questions[qid]['question'],
                         answers=[x['answer'] for x in a['answers']],
                         multiple_choice_answer=a['multiple_choice_answer'],
                         local_image_path=name, image_sha256=sha(args.images / name)))
        if n % 100 == 0:
            print('prepared', n, '/', need, flush=True)
    manifest = dict(
        dataset='VQAv2 val counting',
        selection=__doc__.strip(),
        seed=args.seed, eligible_images=len(ranked), skipped_non_integer_answers=skipped_non_integer,
        excluded_images=len(excluded),
        sources={'questions_zip_sha256': sha(args.questions_zip), 'annotations_zip_sha256': sha(args.annotations_zip),
                 'image_url': COCO_URL.format('<file>')},
        evaluation=rows[:args.count], preflight=rows[args.count:])
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print('written', args.output, len(manifest['evaluation']), len(manifest['preflight']))


if __name__ == '__main__':
    main()
