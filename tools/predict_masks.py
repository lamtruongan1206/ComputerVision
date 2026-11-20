#!/usr/bin/env python
"""Run a trained model on a folder of images and save class-id PNG masks."""

import argparse
import sys
from pathlib import Path

import numpy as np
from mmengine import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agri_pidnet.checkpoint import load_compatible_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    model = init_model(cfg, checkpoint=None, device=args.device)
    load_compatible_checkpoint(model, args.checkpoint)
    paths = sorted(q for q in args.images.iterdir()
                   if q.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    for path in paths:
        pred = inference_model(model, str(path)).pred_sem_seg.data.squeeze(0)
        Image.fromarray(pred.cpu().numpy().astype(np.uint8)).save(
            args.output_dir / f'{path.stem}.png')
    print(f'saved {len(paths)} masks to {args.output_dir}')


if __name__ == '__main__':
    main()
