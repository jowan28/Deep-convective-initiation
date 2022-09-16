#!/bin/bash

#SBATCH --partition=short-serial
#SBATCH --job-name=myjob
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=20:00:00

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"
#module add jaspy
sleep 3

### Provide path to yout python/conda virtual enviornment
### or you can try
conda activate jasenv

PROGRAM="area_cross_evaluation.py"
### DOMAIN can be "sea" or "land"
DOMAIN="sea"
python ${PROGRAM} ${DOMAIN}

BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"

