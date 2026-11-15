"""camera mounting files (common.load_tf_static): yaml static transforms chained to base_link, json passthrough."""

import json

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from hdlabel.common import load_tf_static, tf_to_base


def test_yaml_chain(tmp_path):
    q = R.from_euler("z", 90, degrees=True).as_quat().tolist()
    f = tmp_path / "tf.yaml"
    f.write_text(
        yaml.safe_dump(
            [
                dict(frame="sensor_kit", parent="base_link", translation=[1.0, 0.0, 2.0], rotation=q),
                dict(frame="cam", parent="sensor_kit", translation=[0.5, 0.0, 0.0], rotation=[0, 0, 0, 1]),
                dict(
                    frame="other", parent="odom", translation=[0, 0, 0], rotation=[0, 0, 0, 1]
                ),  # not chained: dropped
            ]
        )
    )
    tf = load_tf_static(str(f))
    assert set(tf) == {"sensor_kit", "cam"}
    T = np.array(tf["cam"])
    assert np.allclose(T[:3, 3], [1.0, 0.5, 2.0])  # 0.5 m along the rotated x axis (= base y)
    j = tmp_path / "tf.json"
    j.write_text(json.dumps(tf))
    assert load_tf_static(str(j)) == tf


def test_tf_to_base_identity():
    assert tf_to_base({"a": ("base_link", np.eye(4))})["a"] == np.eye(4).tolist()
