# Crop-row recognition and robot guidance line (cotton field, Agri-PIDNet-S)

Course project (Computer Vision, HVKTQS): semantic segmentation of cotton field images
into 4 classes and a guidance centerline for an agricultural robot.

| Class | Meaning |
| --- | --- |
| 0 | background |
| 1 | crop_row |
| 2 | furrow |
| 3 | obstacle |

## Credits

The model (Agri-PIDNet-S), training code and dataset come from the original authors:
https://github.com/xyl-design/xinyanlin . This repository contains our reproduction
experiments, the shadow fine-tune configuration, and the centerline tools in `tools/`.
Please follow the license of the original repository and cite the original paper.

## Results (test set, 107 images, `tools/evaluate.py`, mIoU %)

| Model | Seed 0 | Seed 1 | Seed 3407 | Mean ± std |
| --- | --- | --- | --- | --- |
| Baseline (PIDNet-S) | 85.15 | 85.55 | 84.95 | 85.22 ± 0.31 |
| Full Agri-PIDNet-S | 86.02 | 86.32 | 85.57 | 85.97 ± 0.38 |
| Full + shadow fine-tune (seed 1, +4000 iters) | | 86.13 | | 86.13 |

Params / FLOPs: baseline 7.643 M / 23.727 G, full 8.067 M / 20.130 G.

Training setup used here: 10k iterations, batch size 2, no offline augmentation
(the paper uses 20k iterations, batch 4, `cotton_two_stage`; we use it as a reference only).

## Centerline

`tools/centerline_core.py` works on the predicted 512x512 mask:

1. skip if obstacle (class 3) covers at least 3% of the middle 40% width of the lower half;
2. track the furrow bottom-up and take the midpoints;
3. fit a line with exhaustive-pair RANSAC + polyfit, require at least 15 points;
4. draw a thin yellow line about half the image long, stopping 8 px before an obstacle.

Colours (BGR output): crop row green, furrow blue, obstacle red, centerline yellow.

## Setup

Python 3.10 with torch, mmengine, mmcv (lite is enough for training), mmsegmentation 1.2.2
(see the original repository). Dataset: `data/cotton_709/{train,val,test}/{images,masks}`
(709 images split 495 / 107 / 107, from the original authors).

## Train

Stage 1 (10 000 iterations, batch 2), `--experiment` is `baseline` (PIDNet-S) or `full` (Agri-PIDNet-S):

```
python tools/train.py --config configs/exp.py --experiment full --seed 1 \
    --data-root data/cotton_709 --work-dir work_dirs/full_s1
```

`run_seeds.sh` runs baseline and full for seeds 1 and 3407 in a loop (seed 0 uses the same command).

Stage 2, shadow fine-tune (4 000 iterations from the stage-1 seed-1 checkpoint, RandomShadow + PhotoMetricDistortion):

```
python tools/train.py --config configs/ft_shadow.py --experiment full --seed 1 \
    --data-root data/cotton_709 --work-dir work_dirs/shadow_s1 \
    --load-from work_dirs/full_s1/iter_10000.pth
```

`run_shadow.py` is the same command wrapped with a GPU-memory wait for the shared server.
Training logs are in `logs/`. The checkpoint used for results is the last iteration of each stage.

## Evaluate (test set, mIoU)

```
python tools/evaluate.py --config ft_shadow_cfg.py --checkpoint work_dirs/shadow_s1/iter_4000.pth \
    --images data/cotton_709/test/images --masks data/cotton_709/test/masks \
    --output-dir outputs/check_shadow --device cuda:0
# Evaluated 107 images.  mIoU: 86.13%
```

## Inference and centerline

```
python tools/predict_masks.py --config ft_shadow_cfg.py --checkpoint work_dirs/shadow_s1/iter_4000.pth \
       --images test_images --output-dir out/masks --device cuda:0

python tools/centerline_core.py --pred-dir out/masks --images test_images --output-dir out/result
```

Checkpoint (about 98 MB, not stored in git): https://drive.google.com/file/d/1amPOJJKRpx1QVN1EBrleHzsOQFGj0Gr8/view?usp=sharing

## Known limitations

- Low-angle views with tall crops, deep perspective, sparse seedlings, strong shadows.
- No ground-truth centerline, so the line error is not measured; centerlines are judged by eye
  on extra field images outside the dataset (results in the report, Figure 2).
- Not tested on a real robot.
- 3 seeds only; the gap to the baseline (0.75 mIoU) is about twice the seed std.
