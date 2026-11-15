# Environment

Linux with an NVIDIA GPU (CUDA) for real data; the unit tests and the synthetic demo also run on CPU.

## uv (reproducible, pinned)
```bash
git clone https://github.com/yukkysaito/HD-Map-Auto-Labeling.git && cd HD-Map-Auto-Labeling
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r env/requirements.lock.txt --override env/uv-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]" --no-deps
```
- `uv-overrides.txt`: pins torch / torchvision to the CUDA 12.8 builds (needed for RTX 50 series = sm_120).
- `requirements.lock.txt`: output of `uv pip freeze` (2026-11-14).

## pip (any environment)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the build for your CUDA / cpu
pip install -e ".[dev]"
```
`./autolabel` uses `.venv/bin/python` when it exists, otherwise `python3`.

## Docker
```bash
docker build -f docker/Dockerfile -t hdlabel .
docker run --gpus all -v /data:/data -v $PWD/output:/app/output hdlabel run --map /data/map.pcd --logs "/data/logs/*"
```

## Development
`./autolabel test` (pytest), `ruff check . && ruff format --check .`, `pyflakes hdlabel tests`. CI
(`.github/workflows/ci.yml`) runs the same on CPU for every push.
