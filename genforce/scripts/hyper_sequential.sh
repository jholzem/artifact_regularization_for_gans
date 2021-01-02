#!/bin/bash

LAMBDAS=$1
METRIC=$2
LRS=$3
NETHZ=$4

PORT=${PORT:-29500}

GPUS=1
CONFIG=configs/stylegan_ffhq256_fourier_regularized.py

BAR="_"
FOLDER="/cluster/scratch/"
RES="/results/"
ENDING="_generator.pth"
TEMP="_workdir"


for LAMBDA in $LAMBDAS;

do

    for LR in $LRS;

    do

        SAVENAME="${LAMBDA}$BAR${METRIC}$BAR${LR}"


        SYNFOLDER="$FOLDER${NETHZ}$RES$SAVENAME"
        WORK_DIR="$FOLDER${NETHZ}$RES$SAVENAME$TEMP"


        python -m torch.distributed.launch \
               --nproc_per_node=${GPUS} \
               --master_port=${PORT} \
               ./train.py ${CONFIG} \
                   --work_dir ${WORK_DIR} \
                   --launcher="pytorch" \
                   --lamb=${LAMBDA} \
                   --metric=${METRIC} \
                   --baseLR=${LR} \
                   --nethz=${NETHZ}
               ${@:8}


        for idx in {1..9};

        do

            python img_syn.py 10000 "$FOLDER${NETHZ}$RES$SAVENAME$BAR$idx$ENDING" ${SYNFOLDER}
            python demo_dir.py -d ${SYNFOLDER} -m weights/blur_jpg_prob0.1.pth

            rm -r ${SYNFOLDER}

        done

    done

done
