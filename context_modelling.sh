#!/usr/bin/env bash
#!/bin/sh

#SBATCH --time=4:00:00
#SBATCH --partition=nixos
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000
#SBATCH --job-name rep_delieverable
#SBATCH --output delievarble_rep.out

for embeddings in baselines l1 l2 l4 l8 l_unbounded lr1 lr2 lr4 lr8 lr_unbounded r1 r2 r4 r8 r_unbounded
do
# LSTM + Transformer

srun --unbuffered nix develop ../CustomDeepLearningGPU --command python3 src/training_cont.py --config_file continuous_params.yaml --config_name baseline_lstm_transformer --result_csv baselines_cont.csv --embeddings $embeddings


# LSTM

srun --unbuffered nix develop ../CustomDeepLearningGPU --command python3 src/training_cont.py --config_file continuous_params.yaml --config_name baseline_lstm --result_csv baselines_cont.csv --embeddings $embeddings


done
