# GPU image: docker build -f docker/Dockerfile -t hdlabel .
#            docker run --gpus all -v /data:/data -v $PWD/output:/app/output hdlabel run --map /data/map.pcd --logs "/data/logs/*"
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
WORKDIR /app
COPY env/ env/
RUN uv venv .venv --python 3.11 \
    && uv pip install --python .venv/bin/python -r env/requirements.lock.txt --override env/uv-overrides.txt \
       --extra-index-url https://download.pytorch.org/whl/cu128
COPY . .
ENTRYPOINT ["./autolabel"]
CMD ["-h"]
