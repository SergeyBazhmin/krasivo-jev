docker run -d --gpus all \
  -p 8080:8080 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:gemma4 \
  --model google/gemma-4-26B-A4B-it \
  --port 8080 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16000 \
