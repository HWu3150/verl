# Minimal GRPO + vLLM recipe for Qwen3-1.7B on custom JSON dataset
# Adjust paths/lr/batch sizes to your hardware. Tested config is conservative.

set -x
ENGINE=${1:-vllm}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_API_KEY="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"
export RAY_TMPDIR="/work/ray_tmp"
export HF_HOME="/datashare/huggingface/huggingface"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/rewarder_alfworld/train.json \
    data.val_files=./data/rewarder_alfworld/val.json \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path=./custom_reward_openai.py \
    reward.custom_reward_function.reward_kwargs.base_url=https://api.bltcy.ai/v1 \
    reward.custom_reward_function.reward_kwargs.model=qwen3-32b \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_qwen3_1p7b_alfworld' \
    trainer.experiment_name='qwen3_1p7b_vllm_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    \"$@\"
