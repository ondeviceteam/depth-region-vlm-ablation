#!/usr/bin/env python3
"""Run every method and merge strength on one shard of a manifest.

One worker loads one encoder variant of the official ToSA LLaVA-OneVision path
for the model fixed in the experiment's protocol.json: `2D` (official ToMe encoder) serves Token merge 없음, ToMe, Depth-only and
Depth-RGB; `3D` (official ToSA encoder) serves ToSA. Rows are appended to a
jsonl file and finished (condition, question) pairs are skipped on restart.
Scoring is done afterwards by summarize.py.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def merge_r(keep):
    # Unmodified official ToSA/ToMe formula (pixel area / 14^2, divided by 27 config layers).
    return int((384 * 384 / 14 ** 2) * (1 - keep) / 27)


def conditions(variant, dataset, keeps, repeat_keep, repeats):
    out = []
    methods = ['Dense', 'ToMe', 'Depth-only', 'Depth-RGB'] if variant == '2D' else ['ToSA']
    for method in methods:
        for keep in ([1.0] if method == 'Dense' else keeps):
            runs = repeats if (dataset == 'vqav2_count' and (keep == repeat_keep or method == 'Dense')) else 1
            for run in range(runs):
                out.append((method, keep, run))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, help='directory name under experiments/')
    ap.add_argument('--dataset', choices=['vqav2_count', 'gqa'], required=True)
    ap.add_argument('--variant', choices=['2D', '3D'], required=True)
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
    spec = protocol['datasets'][args.dataset]
    assert sha(ROOT / spec['manifest']) == spec['manifest_sha256']
    for rel, digest in protocol['harness_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in protocol['upstream_python_sha256'].items():
        assert sha(args.upstream / rel) == digest, rel
    for rel, digest in protocol['source_python_sha256'].items():
        assert sha(args.source_repo / rel) == digest, rel
    manifest = json.loads((ROOT / spec['manifest']).read_text())
    keeps, repeats, repeat_keep = protocol['keep_settings'], protocol['repeats'], protocol['repeat_keep']
    todo_conditions = conditions(args.variant, args.dataset, keeps, repeat_keep, repeats)
    questions = manifest['evaluation'][args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.dataset}_{args.variant}_shard{args.shard}of{args.num_shards}'
    out_path = args.output / f'{tag}.jsonl'
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r['method'], r['nominal_keep_ratio'], r['run'], str(r['question_id'])))

    sys.path[:0] = [str(args.source_repo / 'src'), str(ROOT / 'scripts'), str(args.upstream / 'ToSA'), str(args.upstream)]
    from depth_official_bridge import DepthBridge, build_single_geometry
    module_spec = importlib.util.spec_from_file_location('official_vqa', args.upstream / 'ToSA/run_vqa_3d_tome.py')
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    models = json.loads((args.runtime / 'models.json').read_text())
    assert models[protocol['model']]['revision'] == protocol['model_revision']
    assert models[protocol['vision_model']]['revision'] == protocol['vision_revision']
    torch.manual_seed(42)
    np.random.seed(42)
    overrides = dict(image_aspect_ratio=None, tome=args.variant, visual_token_merge_ratio=1.,
                     mm_vision_tower=protocol['vision_model'])
    if args.variant == '3D':
        overrides['alpha'] = [i / 25 for i in range(26)]
    tokenizer, model, processor, max_length = official.load_pretrained_model(
        models[protocol['model']]['path'], None, 'llava_qwen', device_map='cuda:0', multimodal=True,
        overwrite_config=overrides, attn_implementation='eager')
    model.eval()
    official.device = 'cuda:0'
    official.tokenizer = tokenizer
    bot = official.llava_chatbot(model, tokenizer, processor, max_length)
    tower = model.get_vision_tower()
    vision = tower.vision_tower
    # FP16 Qwen2 produced NaN in the last decoder layer; keep official FP16 vision/matching, LM/projector BF16.
    model.to(dtype=torch.bfloat16)
    vision.to(dtype=torch.float16)
    model.get_model().mm_projector.register_forward_pre_hook(lambda m, a: tuple(x.to(torch.bfloat16) for x in a))
    finite = [0]

    def check_logits(module, values, result):
        assert bool(torch.isfinite(result).all()), 'Nonfinite language logits'
        finite[0] += 1
    model.lm_head.register_forward_hook(check_logits)
    assert len(vision.vision_model.encoder.layers) == 26
    bridge = DepthBridge(vision, 'Depth-only') if args.variant == '2D' else None
    bridge_rgb = None
    captured, layer_tokens = {}, []
    tower.register_forward_hook(lambda m, a, r: captured.update(shape=list(r.shape)))
    for layer in vision.vision_model.encoder.layers:
        layer.register_forward_hook(lambda m, a, r: layer_tokens.append(int(r[0][0].shape[1])))

    cache = {}

    def prepare(q):
        image = Image.open(args.images / q['local_image_path']).convert('RGB')
        pixels = official.process_images([image], processor, model.config)
        bot.img_tensors = [p.to(device='cuda:0', dtype=torch.float16) for p in pixels]
        assert len(bot.img_tensors) == 1 and bot.img_tensors[0].shape == (3, 384, 384)
        bot.image_sizes = [image.size]
        bot.image_masks = None
        stem = Path(q['local_image_path']).stem
        if stem not in cache:
            if args.variant == '3D':
                cache[stem] = torch.stack([bot.process_depth(str(args.depth / f'{stem}.png'))]).to(
                    dtype=torch.float16, device='cuda:0')
            else:
                cache[stem] = build_single_geometry(np.load(args.depth / f'{stem}.npy'))[0]
        if args.variant == '3D':
            bot.depth_tensors = cache[stem]
            return None
        bot.depth_tensors = None
        return cache[stem]

    def run_condition(method, keep, geometry, question):
        r = 0 if method == 'Dense' else merge_r(keep)
        target = 729 - 26 * r
        vision.visual_token_merge_ratio = keep
        active = None
        if bridge is not None:
            bridge.enabled = False
            if method in ('Depth-only', 'Depth-RGB'):
                bridge.method = method
                feasibility = bridge.configure(geometry, r)
                if feasibility.optimistic_min_tokens > target:
                    bridge.enabled = False
                    return dict(status='STRUCTURAL_INFEASIBLE', raw_response='', generated_tokens=0,
                                actual_tokens=None, protected_tokens=feasibility.protected_singletons,
                                min_tokens=feasibility.optimistic_min_tokens), r, target
                active = feasibility
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
            if bridge is not None:
                bridge.enabled = False
        assert finite[0] > before and captured['shape'] == [1, target, 1152], (method, keep, captured)
        assert layer_tokens == [729 - (j + 1) * r for j in range(26)], (method, keep, layer_tokens)
        row = dict(status='PASS', raw_response=answer, generated_tokens=generated['tokens'], actual_tokens=target)
        if active is not None:
            row.update(protected_tokens=active.protected_singletons, min_tokens=active.optimistic_min_tokens)
        return row, r, target

    environment = dict(tag=tag, started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                       gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                       pbs_job_id=os.environ.get('PBS_JOBID'), python=sys.version,
                       packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pillow']},
                       protocol_sha256=sha(protocol_path), conditions=todo_conditions, questions=len(questions))
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')

    # Preflight: token counts for every condition and r=0 identity of the Depth connection.
    with torch.inference_mode():
        q = manifest['preflight'][0]
        geometry = prepare(q)
        pixels = bot.img_tensors[0].unsqueeze(0)
        if bridge is not None:
            vision.visual_token_merge_ratio = 1.
            bridge.method = 'Depth-RGB'
            bridge.configure(geometry, 0)
            bridged = tower(pixels)
            bridge.enabled = False
            plain = tower(pixels)
            assert torch.equal(bridged, plain), 'Depth connection changes r=0 features'
        for method, keep, _ in {(m, k, 0) for m, k, _ in todo_conditions}:
            run_condition(method, keep, geometry, q['question'])
    print('PREFLIGHT_PASS', tag, flush=True)

    count = 0
    with torch.inference_mode(), out_path.open('a') as stream:
        for index, q in enumerate(questions):
            pending = [c for c in todo_conditions if (c[0], c[1], c[2], str(q['question_id'])) not in done]
            if not pending:
                continue
            geometry = prepare(q)
            for method, keep, run in pending:
                start = time.perf_counter()
                row, r, target = run_condition(method, keep, geometry, q['question'])
                row = dict(dataset=args.dataset, method=method, nominal_keep_ratio=keep, run=run,
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
