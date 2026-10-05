docker run -d --gpus all \
  -p 8080:8080 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:v0.17.1 \
  --model Qwen/Qwen3.5-2B \
  --port 8080 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16000 \
  --language-model-only