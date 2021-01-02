#!/bin/bash

LAMB=$1
METRIC=$2
BASELR=$3
N_IMAGES=$4
NETHZ=$5
PORT=$RANDOM #{PORT:-29500}

BAR="_"
SAVENAME="${LAMB}$BAR${METRIC}$BAR${BASELR}"
FOLDER="/cluster/scratch/"
RES="/results/"
ENDING="_generator.pth"
TEMP="_workdir"

SYNFOLDER="$FOLDER${NETHZ}$RES$SAVENAME"
WORK_DIR="$FOLDER${NETHZ}$RES$SAVENAME$TEMP"

GPUS=1
CONFIG=genforce/configuration.py

python -m torch.distributed.launch \
       --nproc_per_node=${GPUS} \
       --master_port=${PORT} \
       ./genforce/train.py ${CONFIG} \
           --work_dir ${WORK_DIR} \
           --launcher="pytorch" \
           --lamb=${LAMB} \
           --metric=${METRIC} \
           --baseLR=${BASELR} \
           --nethz=${NETHZ}
	   ${@:8}


python img_syn.py ${N_IMAGES} "$FOLDER${NETHZ}$RES$SAVENAME$BAR$idx$ENDING" ${SYNFOLDER}
python ./CNNDetection/demo_dir.py -d ${SYNFOLDER} -m weights/blur_jpg_prob0.1.pth

rm -r ${SYNFOLDER}
