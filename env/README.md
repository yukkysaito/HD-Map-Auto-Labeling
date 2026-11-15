# Environment

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA driver with CUDA.

```bash
git clone git@github.com:yukkysaito/HD-Map-Auto-Labeling.git && cd HD-Map-Auto-Labeling
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r env/requirements.lock.txt --override env/uv-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```
- `uv-overrides.txt`: pins torch / torchvision to the CUDA 12.8 builds (needed for RTX 50 series = sm_120).
- `requirements.lock.txt`: output of `uv pip freeze` (2026-11-14).
- Development: `./autolabel test` (pytest, unit tests) and `.venv/bin/python -m pyflakes hdlabel tests` (unused imports,
  undefined names). Run both after every change.
