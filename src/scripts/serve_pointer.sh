# serve a folder written by `jev-model export` on http://localhost:8080 (`/pooling`), with the `jev_vllm` plugins;
# extra arguments go to vLLM
set -e
export_dir=$(realpath "${1:?usage: serve_pointer.sh EXPORT_DIR [vllm args...]}")
shift
repo=$(dirname "$(realpath "$0")")/../..
docker build -t krasivo-jev-vllm -f "$repo/docker/vllm.Dockerfile" "$repo"
docker run -d --gpus all \
  -p 8080:8080 \
  --ipc=host \
  -v "$export_dir":/model:ro \
  krasivo-jev-vllm \
  --model /model \
  --served-model-name pointer \
  --runner pooling \
  --port 8080 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 \
  "$@"
