#!/bin/bash

cd ..

# custom config
DATA=/path/to/DATA_ROOT
PYTHON=python
TRAINER=DAPL

DATASET=$1          # name of the dataset (e.g., officehome)
CFG=$2             # config file (e.g., ep25-32)
SOURCE_DOMAIN=$3   # source domain (e.g., art)
TARGET_DOMAINS="$4"  # target domains space-separated (e.g., "clipart product real_world")
T=$5               # temperature
TAU=$6             # pseudo label threshold
U=$7               # coefficient for loss_u
NAME=$8            # job name

# Convert target domains string to array for naming
TARGET_DOMAINS_ARRAY=($TARGET_DOMAINS)
TARGET_DOMAINS_STR=$(IFS=_; echo "${TARGET_DOMAINS_ARRAY[*]}")

for SEED in 1; do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAIN}_to_${TARGET_DOMAINS_STR}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
    
    echo "Run this job and save the output to ${DIR}"
    ${PYTHON} train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --source-domains ${SOURCE_DOMAIN} \
        --target-domains ${TARGET_DOMAINS} \
        --output-dir ${DIR} \
        TRAINER.DAPL.T ${T} \
        TRAINER.DAPL.TAU ${TAU} \
        TRAINER.DAPL.U ${U} &
done

wait  # Wait for all background jobs to complete
