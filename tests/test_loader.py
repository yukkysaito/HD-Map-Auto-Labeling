"""Loader helpers (hdlabel/loader.py)."""

import numpy as np


def test_thin_still_keeps_one_sample_per_interval_while_standing():
    from hdlabel.loader import thin_still

    def s(x, t):
        T = np.eye(4)
        T[0, 3] = x
        return dict(T_map_base=T.tolist(), stamp=int(t * 1e9))

    rs = [s(0, 0), s(0, 0.5), s(0.01, 1.0), s(0, 2.0), s(1.0, 2.5), s(2.0, 3.0)]
    assert [r["stamp"] for r in thin_still(rs, 2.0)] == [0, 2e9, 2.5e9, 3e9]
