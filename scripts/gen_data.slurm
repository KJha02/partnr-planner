#!/bin/bash

#SBATCH --job-name=partnrTrain
#SBATCH --mail-user=kj@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --time=02-00:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner/logs/%x.%j.err
#SBATCH --open-mode=append

# Add requeue flag
#SBATCH --requeue

source ~/.bashrc
conda activate habitat
module load cuda/12.4
module load gcc/10.2.0

cd /mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner

# sh scripts/finetune_planner.sh


# Run the main command and capture its exit status
python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/train.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct

python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct


# If the job failed (non-zero exit code), resubmit it
if [ $EXIT_CODE -ne 0 ]; then
    echo "Job failed with exit code $EXIT_CODE. Requeuing..."
    # Submit the same script again
    sbatch $0
    exit $EXIT_CODE
fi
