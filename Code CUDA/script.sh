#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1

./deviceQuery
./firstProgram -N=1024
./firstProgram_grid -N=100000
./addMatrices