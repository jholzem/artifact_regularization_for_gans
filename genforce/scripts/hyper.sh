#!/bin/bash


LAMBDAS="1 5e-1"
METRIC=cos
LRS="1e-3 1e-4"
NETHZ=mschaller


for LAMBDA in $LAMBDAS;

do

    for LR in $LRS;

    do

    bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 10:00 scripts/eval.sh $LAMBDA $METRIC $LR 10000 $NETHZ

    done

done
