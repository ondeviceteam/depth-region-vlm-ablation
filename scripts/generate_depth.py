#!/usr/bin/env python3
"""Depth Anything V2 Small depth for every image in a manifest.

Writes raw float32 relative inverse depth (<stem>.npy, original resolution,
larger = nearer) for the Depth methods, and the 8-bit grayscale PNG
(<stem>.png, per-image min-max to [0, 255]) consumed by the official ToSA code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = sorted({r['local_image_path'] for r in manifest['evaluation'] + manifest['preflight']})
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, cache_dir=args.cache)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION, cache_dir=args.cache).to(args.device).eval()
    records = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        raw, png = args.output / f'{stem}.npy', args.output / f'{stem}.png'
        image = Image.open(args.images / name).convert('RGB')
        if not raw.exists():
            inputs = processor(images=image, return_tensors='pt').to(args.device)
            with torch.inference_mode():
                depth = model(**inputs).predicted_depth
                depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(image.height, image.width),
                                                        mode='bicubic', align_corners=False).squeeze()
            array = depth.float().cpu().numpy().astype(np.float32)
            if array.shape != (image.height, image.width) or not np.isfinite(array).all():
                raise ValueError(f'Invalid depth: {name}')
            temporary = raw.with_suffix('.partial.npy')
            np.save(temporary, array)
            temporary.replace(raw)
        array = np.load(raw)
        assert array.dtype == np.float32 and array.shape == (image.height, image.width) and np.isfinite(array).all()
        if not png.exists():
            span = float(array.max() - array.min())
            unit = (array - array.min()) / span if span > 0 else np.zeros_like(array)
            Image.fromarray((unit * 255).clip(0, 255).astype(np.uint8)).save(png)
        records.append(dict(image=name, raw_sha256=sha(raw), png_sha256=sha(png), shape=list(array.shape)))
        if index % 100 == 0:
            print(f'depth {index + 1}/{len(names)}', flush=True)
    (args.output / 'depth_manifest.json').write_text(json.dumps(dict(
        model=MODEL, revision=REVISION, raw='float32 relative inverse depth, larger is nearer, bicubic to RGB size',
        png='per-image min-max to [0,255], uint8 truncation, grayscale', rows=records), indent=1) + '\n')
    print('done', len(records))


if __name__ == '__main__':
    main()
