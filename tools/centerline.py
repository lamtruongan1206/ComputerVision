#!/usr/bin/env python
"""Guidance centerline from a 4-class mask (0 bg, 1 crop_row, 2 furrow, 3 obstacle).

For each mask: pick the furrow the robot is in (the component nearest the
bottom-centre of the image), take the midpoint of that furrow on scanlines,
fit a line, and report heading angle, lateral offset and an obstacle warning.
Run on predicted masks and on GT masks, then compare the two.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORK = 512
FURROW, OBSTACLE = 2, 3
# BGR colours: crop_row green, furrow (aisle) blue, obstacle red; background untinted
COLORS = {1: (60, 200, 60), 2: (255, 120, 0), 3: (0, 0, 255)}
NAMES = {1: 'crop_row', 2: 'furrow (aisle)', 3: 'obstacle'}
# tunable parameters of the 'center' method (see tools/tune_center.py)
CENTER = dict(side=16, tol=0.02, min_pts=8, min_span=0.15, min_width=6, bottom=0.70)
LINE_W = 2          # centerline thickness in pixels


def load_mask(path):
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return cv2.resize(mask, (WORK, WORK), interpolation=cv2.INTER_NEAREST)


def _runs(row):
    xs = np.nonzero(row)[0]
    if xs.size == 0:
        return []
    return [(p[0], p[-1]) for p in np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)]


def _furrow_points(mask, min_area, step, min_width):
    """'furrow' method: midpoints of the furrow component nearest the robot."""
    h, w = mask.shape
    furrow = (mask == FURROW).astype(np.uint8)
    furrow = cv2.morphologyEx(furrow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(furrow, connectivity=8)
    best, best_d = None, 1e18
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area * h * w:
            continue
        ys, xs = np.nonzero(lab == i)
        d = np.min((ys - (h - 1)) ** 2 + (xs - w / 2) ** 2)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return []
    comp = lab == best
    rows = np.nonzero(comp.any(axis=1))[0]
    y_bot, y_top = int(rows.max()), int(rows.min())
    pts, prev_x = [], None
    for y in range(y_bot, y_top - 1, -step):
        runs = [r for r in _runs(comp[y]) if r[1] - r[0] + 1 >= min_width]
        if not runs:
            continue
        ref = w / 2 if prev_x is None else prev_x
        x1, x2 = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - ref))
        prev_x = (x1 + x2) / 2
        pts.append((prev_x, y, x2 - x1 + 1))
    return pts


def _aisle_points(mask, step, min_width, side=8, max_width=0.5):
    """'aisle' method: a furrow run counts only if crop row lies on BOTH sides.

    This rejects wide open-soil blobs (e.g. in front of an irrigation pipe)
    whose midpoint is meaningless, and tracks the aisle the robot is in.
    """
    h, w = mask.shape
    furrow = (mask == FURROW).astype(np.uint8)
    furrow = cv2.morphologyEx(furrow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    furrow, crop = furrow.astype(bool), (mask == 1)
    pts, prev_x, last_y = [], None, None
    for y in range(h - 1, -1, -step):
        cand = []
        for x1, x2 in _runs(furrow[y]):
            width = x2 - x1 + 1
            if width < min_width or width > max_width * w:
                continue
            if crop[y, max(0, x1 - side):x1].any() and crop[y, x2 + 1:x2 + 1 + side].any():
                cand.append((x1, x2))
        if not cand:
            continue
        ref = w / 2 if prev_x is None else prev_x
        x1, x2 = min(cand, key=lambda r: abs((r[0] + r[1]) / 2 - ref))
        cx = (x1 + x2) / 2
        if prev_x is not None and abs(cx - prev_x) > 0.08 * w + 0.5 * (last_y - y):
            continue
        pts.append((cx, y, x2 - x1 + 1))
        prev_x, last_y = cx, y
    return pts


def _center_points(mask, step, min_width, min_area=0.004, max_width=1.0, side=8):
    """'center' method: pick the furrow nearest the image CENTRE, then take the
    midpoint of its leftmost and rightmost pixel on every scanline.  A scanline
    only counts if crop row lies just outside BOTH ends (it is a real aisle)."""
    h, w = mask.shape
    furrow = (mask == FURROW).astype(np.uint8)
    furrow = cv2.morphologyEx(furrow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(furrow, connectivity=8)
    best, best_d = None, 1e18
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area * h * w:
            continue
        ys, xs = np.nonzero(lab == i)
        d = np.min((ys - h / 2) ** 2 + (xs - w / 2) ** 2)   # 0 if centre inside
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return []
    comp = lab == best
    crop = (mask == 1)
    pts = []
    for y in range(h - 1, -1, -step):
        xs = np.nonzero(comp[y])[0]
        width = xs[-1] - xs[0] + 1 if xs.size else 0
        if width < min_width or width > max_width * w:
            continue
        if not (crop[y, max(0, xs[0] - side):xs[0]].any()
                and crop[y, xs[-1] + 1:xs[-1] + 1 + side].any()):
            continue                                       # no crop row on both sides
        pts.append(((xs[0] + xs[-1]) / 2, y, xs[-1] - xs[0] + 1))
    return pts


def _ransac_line(pts, tol, min_dy):
    """Line x = a*y + b through the MOST points (exhaustive over all pairs, so
    the result is deterministic), then least-squares refit on its inliers."""
    xs, ys = pts[:, 0], pts[:, 1]
    best_n, best_err, best_in = 0, 1e18, None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dy = ys[j] - ys[i]
            if abs(dy) < min_dy:
                continue
            a = (xs[j] - xs[i]) / dy
            b = xs[i] - a * ys[i]
            res = np.abs(xs - (a * ys + b))
            inl = res <= tol
            n_in = int(inl.sum())
            err = float(res[inl].mean())
            if n_in > best_n or (n_in == best_n and err < best_err):
                best_n, best_err, best_in = n_in, err, inl
    if best_in is None or best_n < 3:
        return None
    a, b = np.polyfit(ys[best_in], xs[best_in], 1)
    return float(a), float(b), best_in


def find_centerline(mask, min_area=0.004, step=4, min_width=6, margin=1.0,
                    obs_min=150, method='furrow'):
    """Return a dict describing the centerline, or {'found': False}."""
    h, w = mask.shape
    if method == 'center':
        pts = _center_points(mask, step, CENTER['min_width'], side=CENTER['side'])
        min_pts, min_span, obs_extend = CENTER['min_pts'], CENTER['min_span'] * h, 0
    elif method == 'aisle':
        pts = _aisle_points(mask, step, min_width)
        min_pts, min_span, obs_extend = 8, 0.2 * h, int(0.15 * h)
    else:
        pts = _furrow_points(mask, min_area, step, min_width)
        min_pts, min_span, obs_extend = 5, 0, 0
    if len(pts) < min_pts:
        return {'found': False}
    y_bot, y_top = int(max(p[1] for p in pts)), int(min(p[1] for p in pts))
    if y_bot - y_top < min_span:
        return {'found': False}

    arr = np.array(pts, dtype=np.float32)
    if method == 'center':
        fit = _ransac_line(arr, tol=CENTER['tol'] * w, min_dy=0.1 * h)
        if fit is None:
            return {'found': False}
        a, b, inl = fit                      # x = a*y + b
        arr = arr[inl]                       # keep only the points on the line
        y_bot, y_top = int(arr[:, 1].max()), int(arr[:, 1].min())
        # robot-like view only: the aisle must reach the bottom (near) part of the image
        if y_bot < CENTER['bottom'] * h or y_bot - y_top < CENTER['min_span'] * h:
            return {'found': False}
    else:
        vx, vy, x0, y0 = cv2.fitLine(arr[:, :2], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if abs(vy) < 1e-3:
            return {'found': False}
        a = float(vx / vy)                   # x = a*y + b
        b = float(x0 - a * y0)
    x_bot = a * (h - 1) + b
    x_look = a * (0.6 * h) + b

    # obstacle inside the furrow corridor ahead of the robot
    order = np.argsort(arr[:, 1])
    ys_s, wd_s = arr[order, 1], arr[order, 2]
    oy, ox = np.nonzero(mask == OBSTACLE)
    inside = (oy >= y_top - obs_extend) & (oy <= y_bot)
    oy, ox = oy[inside], ox[inside]
    half = np.interp(oy, ys_s, wd_s) / 2 * margin
    hit = np.abs(ox - (a * oy + b)) <= half
    n_obs = int(hit.sum())
    warn = n_obs >= obs_min
    obs_dist = float((h - 1 - oy[hit].max()) / h) if n_obs else float('nan')

    return {
        'found': True, 'a': a, 'b': b, 'y_top': y_top, 'y_bot': y_bot,
        'angle_deg': -math.degrees(math.atan(a)),   # + = steer right
        'offset_bot': (x_bot - w / 2) / w,           # + = line right of robot
        'offset_look': (x_look - w / 2) / w,
        'n_pts': len(pts), 'pts': arr, 'warn': bool(warn),
        'obs_pixels': n_obs, 'obs_dist': obs_dist,
    }


def lateral_deviation(r1, r2, w=WORK):
    """Mean |x1-x2|/w of two fitted lines over their common row range."""
    top, bot = max(r1['y_top'], r2['y_top']), min(r1['y_bot'], r2['y_bot'])
    if bot <= top:
        return float('nan')
    ys = np.arange(top, bot + 1)
    return float(np.mean(np.abs((r1['a'] - r2['a']) * ys + r1['b'] - r2['b'])) / w)


def path_quality(mask, res, step=8):
    """Quality of a centerline measured on the mask itself (no ground truth).

    imbalance = |dL - dR| / (dL + dR), where dL / dR is the distance from the
    line to the nearest crop-row pixel on its left / right (0 = equidistant).
    in_furrow = share of the line that lies inside the furrow (drivable) region.
    """
    h, w = mask.shape
    crop, furrow = (mask == 1), (mask == FURROW)
    inside, imb = [], []
    for y in range(res['y_top'], res['y_bot'] + 1, step):
        x = int(round(res['a'] * y + res['b']))
        if not 0 <= x < w:
            continue
        inside.append(bool(furrow[y, x]))
        left = np.nonzero(crop[y, :x])[0]
        right = np.nonzero(crop[y, x + 1:])[0]
        if left.size and right.size:
            dl, dr = x - left[-1], right[0] + 1
            imb.append(abs(dl - dr) / (dl + dr))
    return {'in_furrow': float(np.mean(inside)) if inside else float('nan'),
            'imbalance': float(np.mean(imb)) if imb else float('nan')}


def _color_mask(mask):
    out = np.zeros((*mask.shape, 3), np.uint8)
    for k, c in COLORS.items():
        out[mask == k] = c
    return out


def _line(vis, res, color=(0, 255, 255)):
    if res.get('found'):
        y1, y2 = res['y_bot'], res['y_top']
        cv2.line(vis, (int(res['a'] * y1 + res['b']), y1),
                 (int(res['a'] * y2 + res['b']), y2), color, LINE_W, cv2.LINE_AA)


def _title(vis, text, color=(255, 255, 255)):
    cv2.rectangle(vis, (0, 0), (WORK, 24), (0, 0, 0), -1)
    cv2.putText(vis, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)


def draw(image_bgr, pr_mask, pr, gt_mask, gt, out_path):
    """3 panels: image + predicted centerline | predicted mask | ground truth."""
    img = cv2.resize(image_bgr, (WORK, WORK))
    colored = _color_mask(pr_mask)
    tint = (pr_mask > 0)[..., None]
    left = np.where(tint, cv2.addWeighted(img, 0.5, colored, 0.5, 0), img)
    left = left.astype(np.uint8)
    _line(left, pr)
    mid = colored.copy()
    _line(mid, pr)
    right = _color_mask(gt_mask)             # ground truth: mask only, no centerline
    if pr.get('found'):
        text = f"pred: angle {pr['angle_deg']:+.1f} deg, offset {pr['offset_bot']:+.2f}"
        if pr['warn']:
            text += ' OBSTACLE'
    else:
        text = 'pred: no reliable aisle'
    _title(left, text, (0, 0, 255) if (pr.get('found') and pr['warn']) or not pr.get('found')
           else (255, 255, 255))
    _title(mid, 'predicted segmentation + centerline')
    _title(right, 'ground truth segmentation')
    for i, (k, name) in enumerate(NAMES.items()):      # legend on the middle panel
        y = WORK - 14 - 18 * (len(NAMES) - 1 - i)
        cv2.rectangle(mid, (8, y - 10), (22, y + 2), COLORS[k], -1)
        cv2.putText(mid, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.hstack([left, mid, right]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', type=Path, required=True,
                   help='PNG masks predicted by the model (class ids)')
    p.add_argument('--gt-dir', type=Path, required=True,
                   help='ground-truth masks (class ids)')
    p.add_argument('--images', type=Path, default=None,
                   help='original images, only needed for overlays')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--method', choices=('furrow', 'aisle', 'center'), default='furrow')
    p.add_argument('--overlays', type=int, default=12,
                   help='number of overlay images to save (worst first)')
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for gt_path in sorted(args.gt_dir.glob('*.png')):
        pr_path = args.pred_dir / gt_path.name
        if not pr_path.is_file():
            continue
        gt_mask, pr_mask = load_mask(gt_path), load_mask(pr_path)
        g = find_centerline(gt_mask, method=args.method)
        p = find_centerline(pr_mask, method=args.method)
        row = {'image': gt_path.stem, 'found_gt': g['found'],
               'found_pred': p['found']}
        if g['found']:
            row.update(angle_gt=g['angle_deg'], offset_gt=g['offset_bot'],
                       warn_gt=g['warn'])
        if p['found']:
            row.update(angle_pred=p['angle_deg'], offset_pred=p['offset_bot'],
                       warn_pred=p['warn'])
            row.update(path_quality(pr_mask, p))
        if g['found'] and p['found']:
            row['angle_err'] = abs(g['angle_deg'] - p['angle_deg'])
            row['offset_err'] = abs(g['offset_bot'] - p['offset_bot'])
            row['lateral_dev'] = lateral_deviation(g, p)
        rows.append(row)
        cache[gt_path.stem] = (gt_mask, pr_mask, p, g)
    if not rows:
        raise SystemExit('no matching mask pairs found')

    fields = ['image', 'found_gt', 'found_pred', 'angle_gt', 'angle_pred',
              'angle_err', 'offset_gt', 'offset_pred', 'offset_err',
              'lateral_dev', 'warn_gt', 'warn_pred', 'in_furrow', 'imbalance']
    with (args.output_dir / 'centerline_per_image.csv').open(
            'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    both = [r for r in rows if 'angle_err' in r]
    gt_found = [r for r in rows if r['found_gt']]
    false_alarm = sum(1 for r in rows if not r['found_gt'] and r['found_pred'])
    summary = {'images': len(rows),
               'method': args.method,
               'gt_centerline_found': len(gt_found),
               'gt_no_reliable_centerline': len(rows) - len(gt_found),
               'pred_found_without_gt': false_alarm,
               'detection_rate_pct': 100 * len(both) / max(len(gt_found), 1)}
    pf = [r for r in rows if r['found_pred']]
    if pf:
        imb = np.array([r['imbalance'] for r in pf], dtype=float)
        inf = np.array([r['in_furrow'] for r in pf], dtype=float)
        summary.update({
            'pred_found_total': len(pf),
            'pred_found_pct': 100 * len(pf) / len(rows),
            'centering_imbalance_mean': np.nanmean(imb),
            'centering_imbalance_median': np.nanmedian(imb),
            'in_furrow_mean_pct': 100 * np.nanmean(inf),
            'in_furrow_ge90_pct': 100 * np.nanmean(inf >= 0.9)})
    if both:
        ae = np.array([r['angle_err'] for r in both])
        oe = np.array([r['offset_err'] for r in both])
        ld = np.array([r['lateral_dev'] for r in both])
        summary.update({
            'angle_err_mean_deg': ae.mean(), 'angle_err_median_deg': np.median(ae),
            'angle_within_5deg_pct': 100 * (ae <= 5).mean(),
            'angle_within_10deg_pct': 100 * (ae <= 10).mean(),
            'offset_err_mean_width': oe.mean(),
            'offset_within_0.05_pct': 100 * (oe <= 0.05).mean(),
            'lateral_dev_mean_width': np.nanmean(ld)})
    tp = sum(1 for r in both if r['warn_gt'] and r['warn_pred'])
    fp = sum(1 for r in both if not r['warn_gt'] and r['warn_pred'])
    fn = sum(1 for r in both if r['warn_gt'] and not r['warn_pred'])
    summary.update({
        'warn_gt_positive': tp + fn, 'warn_tp': tp, 'warn_fp': fp, 'warn_fn': fn,
        'warn_precision_pct': 100 * tp / (tp + fp) if tp + fp else float('nan'),
        'warn_recall_pct': 100 * tp / (tp + fn) if tp + fn else float('nan')})
    with (args.output_dir / 'centerline_summary.csv').open(
            'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['metric', 'value'])
        for k, v in summary.items():
            wr.writerow([k, f'{v:.4f}' if isinstance(v, float) else v])
    for k, v in summary.items():
        print(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}')

    if args.images and args.overlays > 0:
        ov = args.output_dir / 'overlays'
        ov.mkdir(exist_ok=True)
        ranked = sorted(rows, key=lambda r: -(r.get('angle_err', 1e9)
                                              if r['found_gt'] else -1))
        for r in ranked[:args.overlays]:
            img = None
            for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
                cand = args.images / f"{r['image']}{ext}"
                if cand.is_file():
                    img = cv2.imread(str(cand))
                    break
            if img is None:
                continue
            gt_mask, pr_mask, p, g = cache[r['image']]
            draw(img, pr_mask, p, gt_mask, g, ov / f"{r['image']}.jpg")
        print(f'overlays saved to {ov}')


if __name__ == '__main__':
    main()
