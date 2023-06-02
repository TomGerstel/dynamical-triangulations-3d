#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.gerstel@ru.nl
#SBATCH --partition=hefstud
#SBATCH --output=slurm.txt
#SBATCH --array=0-65%11
#SBATCH --time=4-00:00:00
let i=$(($SLURM_ARRAY_TASK_ID/11))
let j=$(($SLURM_ARRAY_TASK_ID%11))
target/release/dynamical-triangulations-3d -- $1 $i $j
