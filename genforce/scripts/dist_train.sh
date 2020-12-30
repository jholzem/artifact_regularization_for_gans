#!/bin/bash

#GPUS=$1
#CONFIG=$2
#WORK_DIR=$3
PORT=${PORT:-29500}
GPUS=1
CONFIG=configs/stylegan_ffhq256_fourier_regularized.py
WORK_DIR=work_dirs/stylegan_ffhq256_f_reg_train

python -m torch.distributed.launch \
       --nproc_per_node=${GPUS} \
       --master_port=${PORT} \
       ./train.py ${CONFIG} \
           --work_dir ${WORK_DIR} \
           --launcher="pytorch" \
	   ${@:4}
