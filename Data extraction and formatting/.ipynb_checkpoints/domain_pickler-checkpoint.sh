#!/bin/bash

#SBATCH --partition=short-serial
#SBATCH --job-name=myjob
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=10:00:00

### Exit if there is unintialized variable.
#set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"
#module add jaspy
sleep 3

### Provide path to yout python/conda virtual enviornment
### or you can try
conda activate jasenv

### Set input parameters for your python program.


### Insert your the name of python program instead of 'xxx.py'.


PROGRAM="model_pickle.py"
DOMAIN="land"
python ${PROGRAM} ${DOMAIN}

BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"

