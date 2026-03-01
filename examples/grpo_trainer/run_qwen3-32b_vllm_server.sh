export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="/datashare/huggingface/huggingface"

vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 8192 \
  --served-model-name qwen3-32b \
  --trust-remote-code \
  --port 8003