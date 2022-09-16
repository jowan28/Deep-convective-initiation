#!/bin/bash

#SBATCH --partition=short-serial
#SBATCH --job-name=myjob
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=20:00:00
#SBATCH --array=1-2

### Exit if there is unintialized variable.
#set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"
#module add jaspy
sleep 3

### Provide path to yout python/conda virtual enviornment
### or you can try
conda activate jasenv

### Set input parameters for your python program


### Insert your the name of python program instead of 'xxx.py'.

PROGRAM="coefficient_jitters.py"
python ${PROGRAM} ${SLURM_ARRAY_TASK_ID}

BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"


