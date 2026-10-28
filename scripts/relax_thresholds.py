#!/usr/bin/env python3
"""Choose, for every image and merge strength, the boundary threshold of the relaxed Depth methods.

The original Depth methods protect every patch whose depth boundary strength is
>= 0.15 and record STRUCTURAL_INFEASIBLE when protected tokens + regions exceed
the target. The relaxed variant raises the threshold along a fixed ladder
(0.15, 0.20, ..., 1.00, then infinity) and keeps the smallest threshold whose
lower bound fits the target; images that already fit keep 0.15, so their
geometry is identical to the original. If even infinity does not fit (invalid
depth patches and small-region representatives stay protected), the threshold
is null and the condition stays infeasible.

The choice depends only on the depth map, not on the model or the answers.

usage: python scripts/relax_thresholds.py --source-repo DEPTH_REGION_VLM --depth-root DATA_ROOT --output datasets/relaxed_boundary_thresholds.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LADDER = [round(0.15 + 0.05 * i, 2) for i in range(18)] + [float('inf')]
TARGETS = [573, 391, 209, 79]
MANIFESTS = {'vqav2_count': ('datasets/vqav2_count1000.json', 'coco/depth'),
             'gqa': ('datasets/gqa_testdev_balanced.json', 'gqa/depth')}


def resized_depth(path):
    from depth_region_vlm.relative_depth import normalize_relative_inverse_depth
    normalized, _ = normalize_relative_inverse_depth(np.load(path))
    return np.asarray(Image.fromarray(normalized).resize((384, 384), Image.Resampling.BILINEAR), dtype=np.float32)


def geometry(depth, threshold):
    from depth_region_vlm.geometry_v5 import build_geometry_fast
    return build_geometry_fast(depth, 0, 14, 14, k=4, kmeans_iterations=50, min_valid_ratio=.8,
                               boundary_threshold=threshold, small_region_max=4)


def lower_bound(g):
    # Components get labels 0..C-1 and every protected token its own label, so the label count is protected + C.
    return int(g.region.max()) + 1


def choose(path):
    depth = resized_depth(path)
    cache = {}
    out = {}
    for target in TARGETS:
        chosen = None
        for t in LADDER:
            if t not in cache:
                g = geometry(depth, t)
                cache[t] = (lower_bound(g), int(g.protected.sum()))
            if cache[t][0] <= target:
                chosen = t
                break
        key = chosen if chosen is not None else LADDER[-1]
        out[str(target)] = dict(threshold=None if chosen is None else ('inf' if chosen == float('inf') else chosen),
                                lower_bound=cache[key][0], protected_tokens=cache[key][1],
                                original_lower_bound=cache[0.15][0])
    return Path(path).stem, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-repo', type=Path, required=True)
    ap.add_argument('--depth-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source_repo / 'src'))
    result = dict(ladder=[('inf' if t == float('inf') else t) for t in LADDER], targets=TARGETS, images={})
    with Pool(args.workers, initializer=sys.path.insert, initargs=(0, str(args.source_repo / 'src'))) as pool:
        for dataset, (manifest, depth_dir) in MANIFESTS.items():
            data = json.loads((ROOT / manifest).read_text())
            stems = sorted({Path(q['local_image_path']).stem for q in data['evaluation'] + data['preflight']})
            paths = [str(args.depth_root / depth_dir / f'{s}.npy') for s in stems]
            result['images'][dataset] = dict(pool.map(choose, paths, chunksize=4))
            changed = sum(v[str(t)]['threshold'] != 0.15 for v in result['images'][dataset].values() for t in TARGETS)
            print(dataset, len(stems), 'images,', changed, 'image-targets relaxed or infeasible', flush=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
