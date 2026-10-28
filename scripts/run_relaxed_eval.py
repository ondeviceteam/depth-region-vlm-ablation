#!/usr/bin/env python3
"""Run the relaxed Depth methods (boundary threshold chosen per image and merge strength) on one shard.

Same model path, prompt, precision and matching as run_eval.py's Depth-only and
Depth-RGB; the only change is the boundary threshold used to build the depth
geometry, read from datasets/relaxed_boundary_thresholds.json. Rows are
appended to a jsonl file and finished (method, keep, question) triples are
skipped on restart.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'Depth-only-relaxed': 'Depth-only', 'Depth-RGB-relaxed': 'Depth-RGB'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_r(keep):
    return int((384 * 384 / 14 ** 2) * (1 - keep) / 27)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, help='directory name under experiments/')
    ap.add_argument('--model', required=True, help='model key in the protocol, e.g. 0.5B')
    ap.add_argument('--dataset', choices=['vqav2_count', 'gqa'], required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--num-shards', type=int, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--depth', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--source-repo', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol_path = ROOT / 'experiments' / args.experiment / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    model_spec = protocol['models'][args.model]
    spec = protocol['datasets'][args.dataset]
    assert sha(ROOT / spec['manifest']) == spec['manifest_sha256']
    assert sha(ROOT / protocol['thresholds']['file']) == protocol['thresholds']['sha256']
    for rel, digest in protocol['harness_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in protocol['upstream_python_sha256'].items():
        assert sha(args.upstream / rel) == digest, rel
    for rel, digest in protocol['source_python_sha256'].items():
        assert sha(args.source_repo / rel) == digest, rel
    manifest = json.loads((ROOT / spec['manifest']).read_text())
    thresholds = json.loads((ROOT / protocol['thresholds']['file']).read_text())['images'][args.dataset]
    conditions = [(m, k) for m in METHODS for k in protocol['keep_settings']]
    questions = manifest['evaluation'][args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.model}_{args.dataset}_shard{args.shard}of{args.num_shards}'
    out_path = args.output / f'{tag}.jsonl'
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row['method'], row['nominal_keep_ratio'], str(row['question_id'])))

    sys.path[:0] = [str(args.source_repo / 'src'), str(ROOT / 'scripts'), str(args.upstream / 'ToSA'), str(args.upstream)]
    from depth_official_bridge import DepthBridge, build_single_geometry
    from relax_thresholds import geometry, lower_bound, resized_depth
    module_spec = importlib.util.spec_from_file_location('official_vqa', args.upstream / 'ToSA/run_vqa_3d_tome.py')
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    models = json.loads((args.runtime / 'models.json').read_text())
    assert models[model_spec['name']]['revision'] == model_spec['revision']
    assert models[protocol['vision_model']]['revision'] == protocol['vision_revision']
    torch.manual_seed(42)
    np.random.seed(42)
    overrides = dict(image_aspect_ratio=None, tome='2D', visual_token_merge_ratio=1., mm_vision_tower=protocol['vision_model'])
    tokenizer, model, processor, max_length = official.load_pretrained_model(
        models[model_spec['name']]['path'], None, 'llava_qwen', device_map='cuda:0', multimodal=True,
        overwrite_config=overrides, attn_implementation='eager')
    model.eval()
    official.device = 'cuda:0'
    official.tokenizer = tokenizer
    bot = official.llava_chatbot(model, tokenizer, processor, max_length)
    tower = model.get_vision_tower()
    vision = tower.vision_tower
    # Same precision as the original experiments: vision FP16, LM and projector BF16.
    model.to(dtype=torch.bfloat16)
    vision.to(dtype=torch.float16)
    model.get_model().mm_projector.register_forward_pre_hook(lambda m, a: tuple(x.to(torch.bfloat16) for x in a))
    finite = [0]

    def check_logits(module, values, result):
        assert bool(torch.isfinite(result).all()), 'Nonfinite language logits'
        finite[0] += 1
    model.lm_head.register_forward_hook(check_logits)
    assert len(vision.vision_model.encoder.layers) == 26
    bridge = DepthBridge(vision, 'Depth-only')
    captured, layer_tokens = {}, []
    tower.register_forward_hook(lambda m, a, r: captured.update(shape=list(r.shape)))
    for layer in vision.vision_model.encoder.layers:
        layer.register_forward_hook(lambda m, a, r: layer_tokens.append(int(r[0][0].shape[1])))

    depth_cache, geometry_cache = {}, {}

    def prepare(q):
        image = Image.open(args.images / q['local_image_path']).convert('RGB')
        pixels = official.process_images([image], processor, model.config)
        bot.img_tensors = [p.to(device='cuda:0', dtype=torch.float16) for p in pixels]
        assert len(bot.img_tensors) == 1 and bot.img_tensors[0].shape == (3, 384, 384)
        bot.image_sizes = [image.size]
        bot.image_masks = None
        bot.depth_tensors = None
        return Path(q['local_image_path']).stem

    def relaxed_geometry(stem, target):
        choice = thresholds[stem][str(target)]
        if (stem, target) not in geometry_cache:
            if stem not in depth_cache:
                depth_cache[stem] = resized_depth(args.depth / f'{stem}.npy')
            t = choice['threshold']
            g = None if t is None else geometry(depth_cache[stem], float(t))
            assert g is None or lower_bound(g) == choice['lower_bound'], (stem, target)
            geometry_cache[(stem, target)] = g
        return geometry_cache[(stem, target)], choice['threshold']

    def run_condition(method, keep, stem, question):
        r = merge_r(keep)
        target = 729 - 26 * r
        vision.visual_token_merge_ratio = keep
        g, threshold = relaxed_geometry(stem, target)
        extra = dict(boundary_threshold=threshold)
        if g is None:
            return dict(status='STRUCTURAL_INFEASIBLE', raw_response='', generated_tokens=0, actual_tokens=None, **extra), r, target
        bridge.method = METHODS[method]
        feasibility = bridge.configure(g, r)
        assert feasibility.optimistic_min_tokens == lower_bound(g) <= target, (stem, target)
        captured.clear()
        layer_tokens.clear()
        generated = {}
        original_generate = model.generate

        def traced(*a, **kw):
            result = original_generate(*a, **kw)
            generated['tokens'] = int(result.shape[-1])
            return result
        model.generate = traced
        before = finite[0]
        try:
            answer = bot.generate_response(question)
        finally:
            model.generate = original_generate
            bridge.enabled = False
        assert finite[0] > before and captured['shape'] == [1, target, 1152], (method, keep, captured)
        assert layer_tokens == [729 - (j + 1) * r for j in range(26)], (method, keep, layer_tokens)
        return dict(status='PASS', raw_response=answer, generated_tokens=generated['tokens'], actual_tokens=target,
                    protected_tokens=feasibility.protected_singletons, min_tokens=feasibility.optimistic_min_tokens,
                    **extra), r, target

    environment = dict(tag=tag, started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                       gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                       pbs_job_id=os.environ.get('PBS_JOBID'), python=sys.version,
                       packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pillow']},
                       protocol_sha256=sha(protocol_path), conditions=conditions, questions=len(questions))
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')

    # Preflight: at threshold 0.15 the geometry equals the original one, r=0 leaves features unchanged,
    # and every condition runs once with the expected token schedule.
    with torch.inference_mode():
        q = manifest['preflight'][0]
        stem = prepare(q)
        original, _ = build_single_geometry(np.load(args.depth / f'{stem}.npy'))
        same = geometry(resized_depth(args.depth / f'{stem}.npy'), 0.15)
        assert all(np.array_equal(getattr(original, f), getattr(same, f)) for f in ['depth', 'xy', 'region', 'protected'])
        pixels = bot.img_tensors[0].unsqueeze(0)
        vision.visual_token_merge_ratio = 1.
        bridge.method = 'Depth-RGB'
        bridge.configure(original, 0)
        bridged = tower(pixels)
        bridge.enabled = False
        assert torch.equal(bridged, tower(pixels)), 'Depth connection changes r=0 features'
        for method, keep in conditions:
            run_condition(method, keep, stem, q['question'])
    print('PREFLIGHT_PASS', tag, flush=True)

    count = 0
    with torch.inference_mode(), out_path.open('a') as stream:
        for index, q in enumerate(questions):
            pending = [c for c in conditions if (c[0], c[1], str(q['question_id'])) not in done]
            if not pending:
                continue
            stem = prepare(q)
            for method, keep in pending:
                start = time.perf_counter()
                row, r, target = run_condition(method, keep, stem, q['question'])
                row = dict(model=args.model, dataset=args.dataset, method=method, nominal_keep_ratio=keep, run=0,
                           r_per_layer=r, target_tokens=target, question_id=q['question_id'], image_id=q['image_id'],
                           elapsed_seconds=round(time.perf_counter() - start, 4), **row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
            stream.flush()
            if index % 50 == 0:
                print('PROGRESS', tag, index + 1, len(questions), flush=True)
    environment.update(finished_utc=datetime.now(timezone.utc).isoformat(), new_rows=count)
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')
    print('COMPLETE', tag, count, flush=True)


if __name__ == '__main__':
    main()
