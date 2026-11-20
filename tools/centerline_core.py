"""Core of the centerline algorithm (readable version of tools/centerline.py).

Input : mask (H x W) with class ids 0 background, 1 crop_row, 2 furrow, 3 obstacle
Output: (a, b, points) of the line x = a*y + b, or None if no line should be drawn.
"""
import cv2
import numpy as np

FURROW = 2
OBSTACLE = 3
CROP = 1


def midpoints(mask, step=4, min_width=6):
    """Follow the furrow in front of the robot from the bottom of the image upwards.
    Bottom row: the furrow segment containing (or nearest to) the image centre column.
    Each row above: the furrow segment closest to the previous row's midpoint.
    Point = middle of the segment's left and right end."""
    h, w = mask.shape
    furrow = cv2.morphologyEx((mask == FURROW).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    pts, ref = [], w / 2
    for y in range(h - 1, -1, -step):
        d = np.diff(np.concatenate([[0], furrow[y], [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1      # one furrow segment each
        if starts.size == 0:
            continue
        gap = np.maximum(np.maximum(starts - ref, ref - ends), 0)               # 0 if ref is inside
        k = int(np.argmin(gap))
        if ends[k] - starts[k] + 1 >= min_width:
            ref = (starts[k] + ends[k]) / 2
            pts.append((ref, y))
    return np.array(pts, dtype=np.float32)


def line_through_most_points(pts, tol, min_dy):
    """Try every pair of points as a line; keep the one that most points lie on."""
    xs, ys = pts[:, 0], pts[:, 1]
    best_in, best_n = None, 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(ys[j] - ys[i]) < min_dy:
                continue
            a = (xs[j] - xs[i]) / (ys[j] - ys[i])
            b = xs[i] - a * ys[i]
            inliers = np.abs(xs - (a * ys + b)) <= tol
            if inliers.sum() > best_n:
                best_n, best_in = inliers.sum(), inliers
    if best_in is None or best_n < 3:
        return None
    a, b = np.polyfit(ys[best_in], xs[best_in], 1)         # refit on the inliers
    return a, b, pts[best_in]


def obstacle_ahead(mask, thr=0.03):
    """Obstacle (class 3) covering >= 3% of the lower-middle part of the image: the robot cannot go on."""
    h, w = mask.shape
    box = mask[h // 2:, int(0.3 * w):int(0.7 * w)]
    return (box == OBSTACLE).mean() >= thr


def line_ok(mask, a, b, y_top, min_furrow_w=0.04, side=0.30, min_both=0.5, max_on_crop=0.10):
    """Is the line really a path between two crop rows?
    1. it must not run over crop rows (<= 10 % of its points on class 1)
    2. on at least half of the rows there must be crop on BOTH sides of the line (within 30 % of the image width)
    3. the furrow it runs in must be wide enough (median >= 4 % of the image width)"""
    h, w = mask.shape
    on_crop = both = n = 0
    widths = []
    for y in range(h - 1, y_top - 1, -4):
        x = int(round(a * y + b))
        if not 0 <= x < w:
            continue
        n += 1
        on_crop += mask[y, x] == CROP
        r = int(side * w)
        left, right = mask[y, max(0, x - r):x], mask[y, x + 1:x + 1 + r]
        both += bool((left == CROP).any() and (right == CROP).any())
        row = mask[y] == FURROW                              # width of the furrow segment containing x
        l = x
        while l > 0 and row[l - 1]:
            l -= 1
        rr = x
        while rr < w - 1 and row[rr + 1]:
            rr += 1
        widths.append(rr - l + 1)
    if n < 8:
        return False
    return (on_crop / n <= max_on_crop and both / n >= min_both
            and np.median(widths) >= min_furrow_w * w)


def find_centerline(mask, min_points=15):
    h, w = mask.shape
    if obstacle_ahead(mask):                               # obstacle right in front -> no line
        return None
    pts = midpoints(mask)
    pts = pts[pts[:, 1] >= h // 2]                         # only the near half of the image
    if len(pts) < min_points:                              # furrow too small / too short -> no line
        return None
    res = line_through_most_points(pts, tol=0.005 * w, min_dy=0.1 * h)
    if res is None or len(res[2]) < min_points:
        return None
    a, b, inl = res
    if not line_ok(mask, a, b, int(inl[:, 1].min())):      # on crop rows / not between two rows / furrow too narrow
        return None
    return res


# ---------------------------------------------------------------------------
# Command line: save the predicted segmentation and the centerline separately
# ---------------------------------------------------------------------------
WORK = 512                                              # the algorithm runs on 512 x 512 masks
COLORS = {1: (60, 200, 60), 2: (255, 120, 0), 3: (0, 0, 255)}   # BGR: crop_row, furrow, obstacle


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred-dir', type=Path, required=True, help='predicted masks (class ids, PNG)')
    ap.add_argument('--images', type=Path, required=True, help='original images')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    seg_dir, line_dir = args.output_dir / 'segmentation', args.output_dir / 'centerline'
    seg_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    drawn = total = 0
    for path in sorted(args.pred_dir.glob('*.png')):
        image = None
        for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
            if (args.images / (path.stem + ext)).is_file():
                image = cv2.imread(str(args.images / (path.stem + ext)))
                break
        if image is None:
            continue
        total += 1
        H, W = image.shape[:2]
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # 1) predicted segmentation, saved on its own (coloured mask, original size)
        colour = np.zeros((*mask.shape, 3), np.uint8)
        for k, c in COLORS.items():
            colour[mask == k] = c
        colour = cv2.resize(colour, (W, H), interpolation=cv2.INTER_NEAREST)
        # 2) centerline of the furrow in front of the robot, drawn thin on the original image
        small = cv2.resize(mask, (WORK, WORK), interpolation=cv2.INTER_NEAREST)
        res = find_centerline(small)
        out = image.copy()
        if res is not None:
            a, b, _ = res
            y_bot, y_top = WORK - 1, max(int(res[2][:, 1].min()), WORK // 2)
            for y in range(y_bot, y_top, -1):                                   # stop before an obstacle
                x = int(round(a * y + b))
                if 0 <= x < WORK and small[y, x] == OBSTACLE:
                    y_top = y + 8
                    break
            p1 = (int((a * y_bot + b) * W / WORK), int(y_bot * H / WORK))
            p2 = (int((a * y_top + b) * W / WORK), int(y_top * H / WORK))
            cv2.line(out, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(colour, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)   # same line on the segmentation
            cv2.line(colour, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)   # same line on the segmentation
            drawn += 1
        cv2.imwrite(str(line_dir / (path.stem + '.jpg')), out)
        cv2.imwrite(str(seg_dir / (path.stem + '.png')), colour)
        cv2.imwrite(str(seg_dir / (path.stem + '.png')), colour)
    print(f'{drawn}/{total} images have a centerline -> {args.output_dir}')


if __name__ == '__main__':
    main()
