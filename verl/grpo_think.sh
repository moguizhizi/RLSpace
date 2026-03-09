set -x
export CUDA_VISIBLE_DEVICES=6,7
HOME_DIR=$HOME
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


# 1. 清理
ray stop

# 2. 启动 Head
ray start --head --num-gpus 2

# 3. 运行训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME_DIR/workspace/project/RLSpace/verl/mywork/data/train.parquet \
    data.val_files=$HOME_DIR/workspace/project/RLSpace/verl/mywork/data/val.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=2560 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$HOME_DIR/workspace/project/Models/Qwen3-8B-SFT-Merged-TFG/0202 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["swanlab","console"]' \
    trainer.project_name='VeRL-GRPO-Drug-Rec' \
    trainer.experiment_name="qwen3-8b-lora-cos-${TIMESTAMP}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_local_dir=checkpoints \
    trainer.max_actor_ckpt_to_keep=50 \
    trainer.total_epochs=3 \
    reward_model.enable=False \
    custom_reward_function.path=$HOME_DIR/workspace/project/RLSpace/verl/mywork/reward_grpo.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=5120 \