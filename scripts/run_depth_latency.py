#!/usr/bin/env python3
"""Measure latency and FLOPs of the depth model (Depth Anything V2 Small) on the images of a latency protocol sample.

The cost is the same for ToSA and the Depth methods: both start from this depth map. Timed per image
(CUDA synchronized wall clock): preprocess_ms (image processor on CPU and copy to GPU), forward_ms (model),
post_ms (bicubic resize to the image size and copy to CPU). FLOPs: matrix multiplications counted by torch
FlopCounterMode, one pass per image.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from PIL import Image
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol_path = ROOT / 'experiments' / args.experiment / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for rel, digest in protocol['harness_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    depth = protocol['depth']
    images = []
    for dataset, spec in protocol['sample'].items():
        assert sha(ROOT / spec['manifest']) == spec['manifest_sha256']
        names = []
        for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation'][:spec['questions']]:
            if q['local_image_path'] not in names:
                names.append(q['local_image_path'])
        images += [(dataset, args.data / spec['images'] / n) for n in names]
    processor = AutoImageProcessor.from_pretrained(depth['model'], revision=depth['revision'], cache_dir=args.cache)
    model = AutoModelForDepthEstimation.from_pretrained(depth['model'], revision=depth['revision'], cache_dir=args.cache).to('cuda:0').eval()
    args.output.mkdir(parents=True, exist_ok=True)

    def run(path):
        image = Image.open(path).convert('RGB')
        t0 = now()
        inputs = processor(images=image, return_tensors='pt').to('cuda:0')
        t1 = now()
        out = model(**inputs).predicted_depth
        t2 = now()
        out = torch.nn.functional.interpolate(out.unsqueeze(1), size=(image.height, image.width), mode='bicubic',
                                              align_corners=False).squeeze().float().cpu()
        t3 = now()
        return dict(preprocess_ms=round(1000 * (t1 - t0), 3), forward_ms=round(1000 * (t2 - t1), 3), post_ms=round(1000 * (t3 - t2), 3),
                    input_shape=list(inputs['pixel_values'].shape), peak_memory_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1))

    environment = dict(started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       gpu_name=torch.cuda.get_device_name(0), pbs_job_id=os.environ.get('PBS_JOBID'),
                       torch=torch.__version__, protocol_sha256=sha(protocol_path), images=len(images))
    with torch.inference_mode():
        for _, path in images[:protocol['warmup_questions']]:
            run(path)
        with (args.output / 'depth_latency.jsonl').open('w') as stream:
            for repeat in range(protocol['repeats']):
                for dataset, path in images:
                    torch.cuda.reset_peak_memory_stats()
                    stream.write(json.dumps(dict(dataset=dataset, image=path.name, run=repeat, **run(path))) + '\n')
        with (args.output / 'depth_flops.jsonl').open('w') as stream:
            for dataset, path in images:
                image = Image.open(path).convert('RGB')
                inputs = processor(images=image, return_tensors='pt').to('cuda:0')
                with FlopCounterMode(display=False) as counter:
                    model(**inputs)
                stream.write(json.dumps(dict(dataset=dataset, image=path.name, input_shape=list(inputs['pixel_values'].shape),
                                             flops=int(counter.get_total_flops()))) + '\n')
    environment.update(finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output / 'depth_environment.json').write_text(json.dumps(environment, indent=1) + '\n')
    print('COMPLETE depth', len(images), flush=True)


if __name__ == '__main__':
    main()
