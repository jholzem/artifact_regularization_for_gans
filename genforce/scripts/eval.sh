#!/bin/bash

LAMB=$1
METRIC=$2
BASELR=$3
N_IMAGES=$4
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
           --lamb=${LAMB}
           --metric=${METRIC}
           --baselr=${BASELR}
	   ${@:4}

python img_syn.py ${N_IMAGES} 'test_generator.pth'

python demo_dir.py -d test_syn -m weights/blur_jpg_prob0.1.pth

rm -r test_syn
