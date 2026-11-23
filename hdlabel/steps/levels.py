"""Other levels (overpasses, elevated roads, roads below a bridge) around the ego road level of an exported dataset
scene, from the LiDAR point cloud map (hdlabel/levels.py). Optional export addition (config dataset.levels).
Writes SCENE/levels.npz.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.levels import write


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True, help="pointcloud_map.pcd directory (tiles) or a single .pcd file")
    p.add_argument("--scene", required=True, help="dataset scene directory (map.npz)")
    a = p.parse_args(argv)
    print(write(a.scene, a.map))


if __name__ == "__main__":
    main()
