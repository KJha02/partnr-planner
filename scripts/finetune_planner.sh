#!/bin/bash

run_name="finetuning_experiment"
model_path="meta-llama/Llama-3.1-8B-Instruct" # Path to the base model
num_nodes=1  # How many nodes you will be training on
rank=4 # LoRa rank
alpha=128 # LoRa alpha parameter
batch=2
epochs=20
if [  $@ =~ --batched ]; then
    python habitat_llm/finetuning/trainer.py \
    -m \
    wandb.name=$run_name  \
    training_arguments.batch_size=$batch     \
    training_arguments.epochs=$epochs \
    training_arguments.quantize_model=False \
    llm_config.finetune.lora.rank=$rank \
    llm_config.finetune.lora.alpha=$alpha \
    llm_config.name=$model_path \
    dataset.max_train_size=$max_train_size \
    hydra.job.name=$run_name \
    hydra/launcher=slurm_train &

else
    python -m torch.distributed.run --nnodes=$num_nodes --nproc_per_node=$num_nodes \
    habitat_llm/finetuning/trainer.py \
    wandb.name=$run_name  \
    training_arguments.batch_size=$batch \
    training_arguments.epochs=$epochs \
    training_arguments.quantize_model=False \
    llm_config.finetune.lora.rank=$rank \
    llm_config.finetune.lora.alpha=$alpha \
    llm_config.name=$model_path \
    init_distrib=False \
    hydra.job.name=$run_name \

fi
