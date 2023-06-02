#!/bin/bash
#SBATCH --partition=hefstud
#SBATCH --output=slurm.txt
#SBATCH --array=0-3
let i=$(($SLURM_ARRAY_TASK_ID/2))
let j=$(($SLURM_ARRAY_TASK_ID%2))
target/release/dynamical-triangulations-3d -- $1 $i $j
