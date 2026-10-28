#!/usr/bin/env python3
"""Run official ToSA at every fixed merge amount under one quantization setting, on one shard.

One worker loads one model (from the protocol) with one precision setting:
FP (no quantization), or W8A8 / W4A16 fake quantization of either the vision
encoder or the language model (see fake_quant.py). Every question is answered
at every per-layer merge amount r. Rows are appended to a jsonl file and
finished (r, run, question) triples are skipped on restart.
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
PATCHES = 384 * 384 / 14 ** 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keep_for(r):
    # The official encoder computes r = int(PATCHES * (1 - keep) / 27); pick the keep in the middle of r's interval.
    keep = 1.0 if r == 0 else 1 - (r + 0.5) * 27 / PATCHES
    assert int(PATCHES * (1 - keep) / 27) == r
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, help='directory name under experiments/')
    ap.add_argument('--model', required=True, help='model key in the protocol, e.g. 0.5B')
    ap.add_argument('--precision', required=True, help='FP or <vision|llm>-<W8A8|W4A16>')
    ap.add_argument('--dataset', choices=['vqav2_count', 'gqa'], required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--num-shards', type=int, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--depth', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol_path = ROOT / 'experiments' / args.experiment / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert args.precision in protocol['precisions'], args.precision
    model_spec = protocol['models'][args.model]
    spec = protocol['datasets'][args.dataset]
    assert sha(ROOT / spec['manifest']) == spec['manifest_sha256']
    for rel, digest in protocol['harness_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in protocol['upstream_python_sha256'].items():
        assert sha(args.upstream / rel) == digest, rel
    manifest = json.loads((ROOT / spec['manifest']).read_text())
    conditions = [(r, run) for r in protocol['r_values']
                  for run in range(protocol['repeats'] if args.dataset == 'vqav2_count' and r == protocol['repeat_r'] else 1)]
    questions = manifest['evaluation'][args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.model}_{args.precision}_{args.dataset}_shard{args.shard}of{args.num_shards}'
    out_path = args.output / f'{tag}.jsonl'
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row['r_per_layer'], row['run'], str(row['question_id'])))

    sys.path[:0] = [str(ROOT / 'scripts'), str(args.upstream / 'ToSA'), str(args.upstream)]
    import fake_quant
    module_spec = importlib.util.spec_from_file_location('official_vqa', args.upstream / 'ToSA/run_vqa_3d_tome.py')
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    models = json.loads((args.runtime / 'models.json').read_text())
    assert models[model_spec['name']]['revision'] == model_spec['revision']
    assert models[protocol['vision_model']]['revision'] == protocol['vision_revision']
    torch.manual_seed(42)
    np.random.seed(42)
    overrides = dict(image_aspect_ratio=None, tome='3D', visual_token_merge_ratio=1.,
                     mm_vision_tower=protocol['vision_model'], alpha=[i / 25 for i in range(26)])
    tokenizer, model, processor, max_length = official.load_pretrained_model(
        models[model_spec['name']]['path'], None, 'llava_qwen', device_map='cuda:0', multimodal=True,
        overwrite_config=overrides, attn_implementation='eager')
    model.eval()
    official.device = 'cuda:0'
    official.tokenizer = tokenizer
    bot = official.llava_chatbot(model, tokenizer, processor, max_length)
    tower = model.get_vision_tower()
    vision = tower.vision_tower
    # Same precision as the full-precision experiments: vision FP16, LM and projector BF16.
    model.to(dtype=torch.bfloat16)
    vision.to(dtype=torch.float16)
    model.get_model().mm_projector.register_forward_pre_hook(lambda m, a: tuple(x.to(torch.bfloat16) for x in a))
    assert len(vision.vision_model.encoder.layers) == 26

    quantized = 0
    if args.precision != 'FP':
        target, scheme = args.precision.split('-')
        layers = vision.vision_model.encoder.layers if target == 'vision' else model.model.layers
        quantized = fake_quant.apply(layers, scheme)
        group = fake_quant.SCHEMES[scheme]['group']
        probe = next(m for m in layers.modules() if isinstance(m, torch.nn.Linear))
        assert fake_quant.max_levels(probe, group) <= 2 ** fake_quant.SCHEMES[scheme]['weight_bits']
    assert args.precision == 'FP' or quantized == protocol['quantized_linear_layers'][args.model][args.precision.split('-')[0]]

    finite = [0]

    def check_logits(module, values, result):
        assert bool(torch.isfinite(result).all()), 'Nonfinite language logits'
        finite[0] += 1
    model.lm_head.register_forward_hook(check_logits)
    captured, layer_tokens = {}, []
    tower.register_forward_hook(lambda m, a, r: captured.update(shape=list(r.shape)))
    for layer in vision.vision_model.encoder.layers:
        layer.register_forward_hook(lambda m, a, r: layer_tokens.append(int(r[0][0].shape[1])))

    depth_cache = {}

    def prepare(q):
        image = Image.open(args.images / q['local_image_path']).convert('RGB')
        pixels = official.process_images([image], processor, model.config)
        bot.img_tensors = [p.to(device='cuda:0', dtype=torch.float16) for p in pixels]
        assert len(bot.img_tensors) == 1 and bot.img_tensors[0].shape == (3, 384, 384)
        bot.image_sizes = [image.size]
        bot.image_masks = None
        stem = Path(q['local_image_path']).stem
        if stem not in depth_cache:
            depth_cache[stem] = torch.stack([bot.process_depth(str(args.depth / f'{stem}.png'))]).to(
                dtype=torch.float16, device='cuda:0')
        bot.depth_tensors = depth_cache[stem]

    def run_condition(r, question):
        target = 729 - 26 * r
        vision.visual_token_merge_ratio = keep_for(r)
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
        assert finite[0] > before and captured['shape'] == [1, target, 1152], (r, captured)
        assert layer_tokens == [729 - (j + 1) * r for j in range(26)], (r, layer_tokens)
        return dict(status='PASS', raw_response=answer, generated_tokens=generated['tokens'], actual_tokens=target)

    environment = dict(tag=tag, started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                       gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                       pbs_job_id=os.environ.get('PBS_JOBID'), python=sys.version,
                       packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pillow']},
                       protocol_sha256=sha(protocol_path), quantized_linear_layers=quantized,
                       conditions=conditions, questions=len(questions))
    (args.output / f'{tag}_environment.json').write_text(json.dumps(environment, indent=1) + '\n')

    # Preflight: every merge amount runs once with the expected token schedule and finite logits.
    with torch.inference_mode():
        q = manifest['preflight'][0]
        prepare(q)
        for r in protocol['r_values']:
            run_condition(r, q['question'])
    print('PREFLIGHT_PASS', tag, flush=True)

    count = 0
    with torch.inference_mode(), out_path.open('a') as stream:
        for index, q in enumerate(questions):
            pending = [c for c in conditions if (c[0], c[1], str(q['question_id'])) not in done]
            if not pending:
                continue
            prepare(q)
            for r, run in pending:
                start = time.perf_counter()
                row = run_condition(r, q['question'])
                row = dict(model=args.model, precision=args.precision, dataset=args.dataset, method='ToSA',
                           r_per_layer=r, merged_tokens=26 * r, run=run, question_id=q['question_id'],
                           image_id=q['image_id'], elapsed_seconds=round(time.perf_counter() - start, 4), **row)
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
