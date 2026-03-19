import math
import numpy as np
import numba
from numba import njit, prange

# -----------------------------
# Geometry helpers (CPU)
# -----------------------------

@njit(cache=True)
def _rbbox_to_corners(rbbox):
    """
    rbbox: [cx, cy, w, h, angle]
    returns corners as (4,2) clockwise
    """
    cx, cy, w, h, ang = rbbox
    c = math.cos(ang)
    s = math.sin(ang)
    hw = 0.5 * w
    hh = 0.5 * h

    # local corners (clockwise)
    pts = np.empty((4, 2), dtype=np.float32)
    pts[0, 0] = -hw; pts[0, 1] = -hh
    pts[1, 0] = -hw; pts[1, 1] =  hh
    pts[2, 0] =  hw; pts[2, 1] =  hh
    pts[3, 0] =  hw; pts[3, 1] = -hh

    # rotate + translate
    out = np.empty((4, 2), dtype=np.float32)
    for i in range(4):
        x = pts[i, 0]
        y = pts[i, 1]
        out[i, 0] = c * x + s * y + cx
        out[i, 1] = -s * x + c * y + cy
    return out


@njit(cache=True)
def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


@njit(cache=True)
def _inside(px, py, ax, ay, bx, by):
    # Check if point P is inside edge A->B (clockwise polygon)
    # For clockwise, inside is to the RIGHT of edge => cross(B-A, P-A) <= 0
    return _cross(bx - ax, by - ay, px - ax, py - ay) <= 0.0


@njit(cache=True)
def _line_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Intersection between segments AB and CD, assuming they are not parallel
    Returns intersection point (x,y).
    """
    rpx = bx - ax
    rpy = by - ay
    spx = dx - cx
    spy = dy - cy
    denom = _cross(rpx, rpy, spx, spy)
    # caller should ensure denom != 0
    t = _cross(cx - ax, cy - ay, spx, spy) / denom
    return ax + t * rpx, ay + t * rpy


@njit(cache=True)
def _poly_area(poly, n):
    # Shoelace
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return 0.5 * abs(area)


@njit(cache=True)
def _clip_polygon(subject, subject_n, clipper, clipper_n, out):
    """
    Sutherland–Hodgman polygon clipping.
    subject: (maxN,2), subject_n actual vertices
    clipper: (M,2), clipper_n actual vertices (here 4)
    out: (maxN,2) output buffer
    returns out_n
    """
    # We need two buffers to ping-pong
    buf1 = np.empty((32, 2), dtype=np.float32)
    buf2 = np.empty((32, 2), dtype=np.float32)

    # init
    in_n = subject_n
    for i in range(in_n):
        buf1[i, 0] = subject[i, 0]
        buf1[i, 1] = subject[i, 1]

    # clip against each edge of clipper
    for e in range(clipper_n):
        ax = clipper[e, 0]
        ay = clipper[e, 1]
        bx = clipper[(e + 1) % clipper_n, 0]
        by = clipper[(e + 1) % clipper_n, 1]

        if in_n == 0:
            return 0

        out_n = 0
        sx = buf1[in_n - 1, 0]
        sy = buf1[in_n - 1, 1]
        s_inside = _inside(sx, sy, ax, ay, bx, by)

        for i in range(in_n):
            ex = buf1[i, 0]
            ey = buf1[i, 1]
            e_inside = _inside(ex, ey, ax, ay, bx, by)

            if e_inside:
                if not s_inside:
                    # add intersection
                    denom = _cross((ex - sx), (ey - sy), (bx - ax), (by - ay))
                    if abs(denom) > 1e-12:
                        ix, iy = _line_intersect(sx, sy, ex, ey, ax, ay, bx, by)
                        buf2[out_n, 0] = ix
                        buf2[out_n, 1] = iy
                        out_n += 1
                # add endpoint
                buf2[out_n, 0] = ex
                buf2[out_n, 1] = ey
                out_n += 1
            elif s_inside:
                # add intersection
                denom = _cross((ex - sx), (ey - sy), (bx - ax), (by - ay))
                if abs(denom) > 1e-12:
                    ix, iy = _line_intersect(sx, sy, ex, ey, ax, ay, bx, by)
                    buf2[out_n, 0] = ix
                    buf2[out_n, 1] = iy
                    out_n += 1

            sx, sy = ex, ey
            s_inside = e_inside

        # swap buffers
        in_n = out_n
        for i in range(in_n):
            buf1[i, 0] = buf2[i, 0]
            buf1[i, 1] = buf2[i, 1]

    # copy to out
    for i in range(in_n):
        out[i, 0] = buf1[i, 0]
        out[i, 1] = buf1[i, 1]
    return in_n


@njit(cache=True)
def _rotated_iou_single(r1, r2, criterion):
    # areas
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    p1 = _rbbox_to_corners(r1)
    p2 = _rbbox_to_corners(r2)

    inter_poly = np.empty((32, 2), dtype=np.float32)
    inter_n = _clip_polygon(p1, 4, p2, 4, inter_poly)
    inter_area = _poly_area(inter_poly, inter_n)

    if inter_area <= 0.0:
        return 0.0

    if criterion == -1:
        denom = area1 + area2 - inter_area
        return inter_area / denom if denom > 0.0 else 0.0
    elif criterion == 0:
        return inter_area / area1
    elif criterion == 1:
        return inter_area / area2
    else:
        return inter_area


@njit(parallel=True, cache=True)
def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1):
    """
    boxes: (N,5) float32
    query_boxes: (K,5) float32
    returns iou: (N,K) float32
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    out = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return out

    for i in prange(N):
        for j in range(K):
            out[i, j] = _rotated_iou_single(query_boxes[j], boxes[i], criterion)
    return out
