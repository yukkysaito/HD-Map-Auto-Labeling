# 環境の再現

[uv](https://docs.astral.sh/uv/) と CUDA の使える NVIDIA ドライバが必要。

```bash
git clone git@github.com:yukkysaito/HD-Map-Auto-Labeling.git && cd HD-Map-Auto-Labeling
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r env/requirements.lock.txt --override env/uv-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```
- `uv-overrides.txt`: torch / torchvision を CUDA 12.8 版に固定（RTX 50 系 = sm_120 に必要）。
- `requirements.lock.txt`: `uv pip freeze` の結果（2026-11-14）。
- 開発用: `./autolabel test`（pytest、単体テスト）、`.venv/bin/python -m pyflakes hdlabel tests`（未使用 import・未定義名）。変更後に両方実行。
