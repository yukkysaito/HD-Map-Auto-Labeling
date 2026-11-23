"""Shape of road boundary rings (vectorize.py --boundary-shape).

The outline of the drivable area follows the label raster, so it carries the noise of the class border: small
wiggles along a real road edge, and ragged zig-zags where road-coloured cells meet scattered sidewalk / vegetation
cells (plazas, planters, the edge of the view), where there is no clean edge at all. On a ring of the outline (closed,
grid-local xy, metres, sampled every `step` m):
  ragged   per point: back-and-forth turning of the outline within window_m around it, i.e. the total absolute heading
           change of its chords (chord_m long) minus the absolute net heading change, in deg / m. A straight edge, a
           curve or a corner turn one way only (near 0); a zig-zag turns both ways. Points above max_zigzag are not a
           clean boundary (vectorize: observed false).
  smooth   Gaussian smoothing along the ring (sigma smooth_m): the wiggles of a real edge go; a corner of radius r
           moves inwards by about smooth_m**2 / (2 r).
  short    observed runs shorter than min_run_m left between ragged / unknown stretches become unknown too.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def resample_ring(xy: np.ndarray, step: float) -> np.ndarray:
    """closed ring (first point not repeated at the end) -> points every ~step m along it (same orientation)."""
    xy = np.asarray(xy, float)
    q = np.r_[xy, xy[:1]]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    n = max(3, int(round(s[-1] / step)))
    at = np.arange(n) * s[-1] / n
    return np.stack([np.interp(at, s, q[:, k]) for k in (0, 1)], 1)


def smooth_ring(xy: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Gaussian smoothing of a uniformly sampled closed ring (sigma in points)."""
    if sigma_pts <= 0 or len(xy) < 5:
        return np.asarray(xy, float)
    return gaussian_filter1d(np.asarray(xy, float), sigma_pts, axis=0, mode="wrap")


def ring_zigzag(xy: np.ndarray, step: float, window_m: float = 3.0, chord_m: float = 0.5) -> np.ndarray:
    """per point of a uniformly sampled closed ring (spacing step m): back-and-forth turning (deg / m) within window_m
    around it (module doc, `ragged`)."""
    xy = np.asarray(xy, float)
    k = max(1, int(round(chord_m / step)))
    w = max(1, int(round(window_m / step)))
    if len(xy) < 2 * k + 3 or len(xy) <= w:
        return np.zeros(len(xy))
    d = np.roll(xy, -k, axis=0) - xy  # chord from point i to point i + k
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = (np.roll(h, -1) - h + np.pi) % (2 * np.pi) - np.pi
    tv = uniform_filter1d(np.abs(dh), w, mode="wrap") * w
    net = np.abs(uniform_filter1d(dh, w, mode="wrap") * w)
    return np.roll(np.degrees(tv - net) / window_m, k // 2)  # chord i is centred at i + k / 2


def drop_short_runs(flag: np.ndarray, min_pts: int, closed: bool = False) -> np.ndarray:
    """flag with its True runs shorter than min_pts points set False; closed: the sequence is a ring (a run may wrap
    around its end)."""
    out = np.asarray(flag, bool).copy()
    if closed and not out.all() and out.any():
        k = int(np.argmin(out))  # start at a False point: no run wraps
        return np.roll(drop_short_runs(np.roll(out, -k), min_pts), k)
    edges = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_pts:
            out[s:e] = False
    return out
