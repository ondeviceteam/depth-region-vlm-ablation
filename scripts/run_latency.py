#!/usr/bin/env python3
"""Measure latency, peak GPU memory and FLOPs of every method on the fixed question sample of a latency protocol.

One process loads one encoder variant of the official ToSA LLaVA-OneVision path, exactly as run_eval.py and
run_relaxed_eval.py do: `2D` serves No token merge, ToMe, Depth-only and Depth-RGB (per-image threshold from
datasets/relaxed_boundary_thresholds.json); `3D` serves ToSA. The depth model itself is timed by
run_depth_latency.py, because its cost is the same for ToSA and the Depth methods.

Timed per answer (CUDA synchronized wall clock):
  prep_ms       method-specific work from the depth map to the model input
                (ToSA: uint8 depth image + official process_depth + copy to GPU;
                 Depth methods: normalization, 384x384 resize, region geometry, bridge.configure)
  vision_ms     vision tower forward, including the merge steps
  projector_ms  mm_projector forward
  prefill_ms    first language-model forward (all prompt tokens)
  decode_ms     all later language-model forwards
  generate_ms   the whole official generate_response call
Image decoding and process_images are timed once per question as image_ms (same for every method).
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
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_r(keep):
    # Unmodified official ToSA/ToMe formula (pixel area / 14^2, divided by 27 config layers).
    return int((384 * 384 / 14 ** 2) * (1 - keep) / 27)


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def top_flops(counts, prefix):
    # generate() calls the vision tower and projector outside the language model, so FlopCounterMode lists them as
    # top-level modules: SigLipVisionTower2Dtome or ...3Dtome, Sequential (mm_projector) and LlavaQwenForCausalLM.
    names = [k for k in counts if k.startswith(prefix) and '.' not in k]
    assert len(names) == 1, (prefix, names)
    return int(sum(counts[names[0]].values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, help='directory name under experiments/')
    ap.add_argument('--model', required=True, help='model key in the protocol, e.g. 0.5B')
    ap.add_argument('--variant', choices=['2D', '3D'], required=True)
    ap.add_argument('--data', type=Path, required=True, help='root with coco/ and gqa/ images and depth')
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--source-repo', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol_path = ROOT / 'experiments' / args.experiment / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    model_spec = protocol['models'][args.model]
    for rel, digest in protocol['harness_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in protocol['upstream_python_sha256'].items():
        assert sha(args.upstream / rel) == digest, rel
    for rel, digest in protocol['source_python_sha256'].items():
        assert sha(args.source_repo / rel) == digest, rel
    assert sha(ROOT / protocol['thresholds']['file']) == protocol['thresholds']['sha256']
    thresholds = json.loads((ROOT / protocol['thresholds']['file']).read_text())['images']
    sample = []
    for dataset, spec in protocol['sample'].items():
        assert sha(ROOT / spec['manifest']) == spec['manifest_sha256']
        for q in json.loads((ROOT / spec['manifest']).read_text())['evaluation'][:spec['questions']]:
            sample.append(dict(q, dataset=dataset, images=args.data / spec['images'], depth=args.data / spec['depth']))
    keeps = protocol['keep_settings']
    if args.variant == '2D':
        conditions = [('Dense', 1.0)] + [(m, k) for m in ['ToMe', 'Depth-only', 'Depth-RGB'] for k in keeps]
    else:
        conditions = [('ToSA', k) for k in keeps]
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.model}_{args.variant}'

    sys.path[:0] = [str(args.source_repo / 'src'), str(ROOT / 'scripts'), str(args.upstream / 'ToSA'), str(args.upstream)]
    from depth_official_bridge import DepthBridge
    from depth_region_vlm.relative_depth import normalize_relative_inverse_depth
    from relax_thresholds import geometry, lower_bound
    module_spec = importlib.util.spec_from_file_location('official_vqa', args.upstream / 'ToSA/run_vqa_3d_tome.py')
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    models = json.loads((args.runtime / 'models.json').read_text())
    assert models[model_spec['name']]['revision'] == model_spec['revision']
    assert models[protocol['vision_model']]['revision'] == protocol['vision_revision']
    torch.manual_seed(42)
    np.random.seed(42)
    overrides = dict(image_aspect_ratio=None, tome=args.variant, visual_token_merge_ratio=1.,
                     mm_vision_tower=protocol['vision_model'])
    if args.variant == '3D':
        overrides['alpha'] = [i / 25 for i in range(26)]
    tokenizer, model, processor, max_length = official.load_pretrained_model(
        models[model_spec['name']]['path'], None, 'llava_qwen', device_map='cuda:0', multimodal=True,
        overwrite_config=overrides, attn_implementation='eager')
    model.eval()
    official.device = 'cuda:0'
    official.tokenizer = tokenizer
    bot = official.llava_chatbot(model, tokenizer, processor, max_length)
    tower = model.get_vision_tower()
    vision = tower.vision_tower
    # Same precision as the accuracy experiments: vision FP16, LM and projector BF16.
    model.to(dtype=torch.bfloat16)
    vision.to(dtype=torch.float16)
    projector = model.get_model().mm_projector
    projector.register_forward_pre_hook(lambda m, a: tuple(x.to(torch.bfloat16) for x in a))
    assert len(vision.vision_model.encoder.layers) == 26
    bridge = DepthBridge(vision, 'Depth-only') if args.variant == '2D' else None

    # Stage marks. The hooks synchronize, so every stage time is complete GPU work.
    marks = dict(vision=[], projector=[], lm=[])
    shape = {}
    tower.register_forward_pre_hook(lambda m, a: marks['vision'].append([now(), None]))
    def tower_done(module, a, result):
        # A forward hook that returns a value replaces the output, so this one returns nothing.
        marks['vision'][-1][1] = now()
        shape['vision'] = list(result.shape)
    tower.register_forward_hook(tower_done)
    projector.register_forward_pre_hook(lambda m, a: marks['projector'].append([now(), None]))
    projector.register_forward_hook(lambda m, a, r: marks['projector'][-1].__setitem__(1, now()))
    lm = model.get_model()

    def lm_pre(module, a, kw):
        embeds, ids = kw.get('inputs_embeds'), kw.get('input_ids')
        marks['lm'].append([now(), None, int((embeds if embeds is not None else ids).shape[1])])
    lm.register_forward_pre_hook(lm_pre, with_kwargs=True)
    lm.register_forward_hook(lambda m, a, r: marks['lm'][-1].__setitem__(1, now()))

    def prepare_image(q):
        start = now()
        image = Image.open(q['images'] / q['local_image_path']).convert('RGB')
        pixels = official.process_images([image], processor, model.config)
        bot.img_tensors = [p.to(device='cuda:0', dtype=torch.float16) for p in pixels]
        bot.image_sizes = [image.size]
        bot.image_masks = None
        bot.depth_tensors = None
        assert len(bot.img_tensors) == 1 and bot.img_tensors[0].shape == (3, 384, 384)
        return 1000 * (now() - start)

    def prepare_method(method, keep, q, raw):
        # Method-specific work from the depth model output (raw inverse depth) to the model input.
        r = 0 if method == 'Dense' else merge_r(keep)
        target = 729 - 26 * r
        vision.visual_token_merge_ratio = keep
        if bridge is not None:
            bridge.enabled = False
        threshold = None
        start = now()
        if method == 'ToSA':
            span = float(raw.max() - raw.min())
            unit = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
            png = Image.fromarray((unit * 255).clip(0, 255).astype(np.uint8))
            bot.depth_tensors = torch.stack([bot.process_depth(png)]).to(dtype=torch.float16, device='cuda:0')
        elif method in ('Depth-only', 'Depth-RGB'):
            stem = Path(q['local_image_path']).stem
            threshold = thresholds[q['dataset']][stem][str(target)]['threshold']
            normalized, _ = normalize_relative_inverse_depth(raw)
            resized = np.asarray(Image.fromarray(normalized).resize((384, 384), Image.Resampling.BILINEAR), dtype=np.float32)
            g = geometry(resized, float(threshold))
            bridge.method = method
            feasibility = bridge.configure(g, r)
            assert feasibility.optimistic_min_tokens == lower_bound(g) <= target, (stem, target)
        return 1000 * (now() - start), r, target, threshold

    def answer(question):
        for v in marks.values():
            v.clear()
        generated = {}
        original_generate = model.generate

        def traced(*a, **kw):
            result = original_generate(*a, **kw)
            generated['tokens'] = int(result.shape[-1])
            return result
        model.generate = traced
        try:
            start = now()
            bot.generate_response(question)
            total = 1000 * (now() - start)
        finally:
            model.generate = original_generate
            if bridge is not None:
                bridge.enabled = False
        return total, generated['tokens']

    def run(q, raw, method, keep):
        prep_ms, r, target, threshold = prepare_method(method, keep, q, raw)
        torch.cuda.reset_peak_memory_stats()
        generate_ms, tokens = answer(q['question'])
        assert shape['vision'] == [1, target, 1152], (method, keep, shape)
        (v0, v1), (p0, p1) = marks['vision'][0], marks['projector'][0]
        lm_calls = marks['lm']
        prefill = lm_calls[0]
        return dict(method=method, nominal_keep_ratio=keep, r_per_layer=r, target_tokens=target, boundary_threshold=threshold,
                    prep_ms=round(prep_ms, 3), vision_ms=round(1000 * (v1 - v0), 3), projector_ms=round(1000 * (p1 - p0), 3),
                    prefill_ms=round(1000 * (prefill[1] - prefill[0]), 3),
                    decode_ms=round(1000 * sum(c[1] - c[0] for c in lm_calls[1:]), 3), decode_steps=len(lm_calls) - 1,
                    generate_ms=round(generate_ms, 3), llm_input_tokens=prefill[2], vision_output_tokens=shape['vision'][1],
                    generated_tokens=tokens, peak_memory_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1))

    environment = dict(tag=tag, started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                       gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                       pbs_job_id=os.environ.get('PBS_JOBID'), python=sys.version, cpu_threads=torch.get_num_threads(),
                       packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pillow']},
                       protocol_sha256=sha(protocol_path), conditions=conditions, questions=len(sample))
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')

    def raw_depth(q):
        return np.load(q['depth'] / f"{Path(q['local_image_path']).stem}.npy")

    with torch.inference_mode():
        # Warm-up: not recorded.
        for q in sample[:protocol['warmup_questions']]:
            prepare_image(q)
            raw = raw_depth(q)
            for method, keep in conditions:
                run(q, raw, method, keep)
        print('WARMUP_DONE', tag, flush=True)
        # Timed runs: conditions interleaved inside each question so slow drift affects all methods equally.
        with (args.output / f'{tag}_latency.jsonl').open('w') as stream:
            for repeat in range(protocol['repeats']):
                for index, q in enumerate(sample):
                    image_ms = prepare_image(q)
                    raw = raw_depth(q)
                    for method, keep in conditions:
                        row = run(q, raw, method, keep)
                        stream.write(json.dumps(dict(model=args.model, dataset=q['dataset'], question_id=q['question_id'],
                                                     image_id=q['image_id'], run=repeat, image_ms=round(image_ms, 3), **row)) + '\n')
                    stream.flush()
                    if index % 25 == 0:
                        print('PROGRESS', tag, repeat, index + 1, len(sample), flush=True)
        # FLOPs: matrix multiplications counted by torch FlopCounterMode, one pass per question and condition.
        flop_sample = [q for d, spec in protocol['sample'].items()
                       for q in [s for s in sample if s['dataset'] == d][:protocol['flop_questions_per_dataset']]]
        with (args.output / f'{tag}_flops.jsonl').open('w') as stream:
            for q in flop_sample:
                prepare_image(q)
                raw = raw_depth(q)
                for method, keep in conditions:
                    _, r, target, _ = prepare_method(method, keep, q, raw)
                    for v in marks.values():
                        v.clear()
                    with FlopCounterMode(display=False) as counter:
                        bot.generate_response(q['question'])
                    if bridge is not None:
                        bridge.enabled = False
                    counts = counter.get_flop_counts()
                    total = int(counter.get_total_flops())
                    vision_flops, projector_flops = top_flops(counts, 'SigLipVisionTower'), top_flops(counts, 'Sequential')
                    language_flops = top_flops(counts, 'LlavaQwenForCausalLM')
                    stream.write(json.dumps(dict(model=args.model, dataset=q['dataset'], question_id=q['question_id'],
                                                 method=method, nominal_keep_ratio=keep, target_tokens=target,
                                                 total_flops=total, vision_flops=vision_flops, projector_flops=projector_flops,
                                                 language_flops=language_flops, other_flops=total - vision_flops - projector_flops - language_flops,
                                                 llm_input_tokens=marks['lm'][0][2], decode_steps=len(marks['lm']) - 1)) + '\n')
    environment.update(finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')
    print('COMPLETE', tag, flush=True)


if __name__ == '__main__':
    main()
