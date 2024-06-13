#!/bin/bash

#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1 # total number of nodes
##SBATCH --nodelist=

##SBATCH --time=01:00:00   # walltime
#SBATCH -J "ex-dedup"  # job name
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err

# Slurm Variables
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
eval "$(conda shell.bash hook)"
conda activate dedup

# open-platypus
# python $HOME/workspace/bigcode-dataset/decontamination/exact_dedup.py --dataset-path $HOME/workspace/evol-instruct-playground/outputs/merged_datasets/evol-instruct-92a0_merged.jsonl --output-dir $HOME/workspace/evol-instruct-playground/outputs/exact_dedup_datasets/ --ds-output-name open-platypus

# helpsteer
# python $HOME/workspace/bigcode-dataset/decontamination/exact_dedup.py --dataset-path $HOME/workspace/evol-instruct-playground/outputs/merged_datasets/evol-instruct-1500_merged.jsonl --output-dir $HOME/workspace/evol-instruct-playground/outputs/exact_dedup_datasets/ --ds-output-name helpsteer

# WebInstructSub
python $HOME/workspace/bigcode-dataset/decontamination/exact_dedup.py --dataset-path $HOME/workspace/evol-instruct-playground/outputs/merged_datasets/evol-instruct-8533_merged.jsonl --output-dir $HOME/workspace/evol-instruct-playground/outputs/exact_dedup_datasets/ --ds-output-name webinstructsub

# star-aligned-oss
# python $HOME/workspace/bigcode-dataset/decontamination/exact_dedup.py --dataset-path $HOME/workspace/evol-instruct-playground/outputs/merged_datasets/evol-instruct-9903_merged.jsonl --output-dir $HOME/workspace/evol-instruct-playground/outputs/exact_dedup_datasets/ --ds-output-name star-aligned-oss

# code_instructions_alpaca122k
# python $HOME/workspace/bigcode-dataset/decontamination/exact_dedup.py --dataset-path $HOME/workspace/evol-instruct-playground/outputs/merged_datasets//evol-instruct-368d_merged.jsonl --output-dir $HOME/workspace/evol-instruct-playground/outputs/exact_dedup_datasets/ --ds-output-name code_instructions_alpaca122k