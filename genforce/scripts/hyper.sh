#!/bin/bash


LAMBDAS=$1
METRIC=$2
LRS=$3
NETHZ=$4


for LAMBDA in $LAMBDAS;

do

    for LR in $LRS;

    do

    bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 1:00 scripts/eval.sh ${LAMBDA} ${METRIC} ${LR} 10000 ${NETHZ}

    done

done
