#!/bin/bash

LAMB=$1
METRIC=$2
BASELR=$3
N_IMAGES=$4
NETHZ=$5
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
           --lamb=${LAMB} \
           --metric=${METRIC} \
           --baseLR=${BASELR} \
           --nethz=${NETHZ}
	   ${@:8}


for idx in {1..9};

do

BAR="_"
SAVENAME="${LAMB}$BAR${METRIC}$BAR${BASELR}"

FOLDER="/cluster/scratch/"
SLASH="/"
ENDING="_generator.pth"

python img_syn.py ${N_IMAGES} "$FOLDER${NETHZ}$SLASH$SAVENAME$BAR$idx$ENDING"
python demo_dir.py -d test_syn -m weights/blur_jpg_prob0.1.pth

rm -r test_syn

done
