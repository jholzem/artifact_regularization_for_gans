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
CONFIG=configs/stylegan_ffhq256_fourier_regularized.py


idx=1

while true;

do

    FILE="$FOLDER${NETHZ}$RES$SAVENAME$BAR$idx$ENDING"

    if test -f "$FILE"; then

        python img_syn.py ${N_IMAGES} ${FILE} ${SYNFOLDER}
        python demo_dir.py -d ${SYNFOLDER} -m weights/blur_jpg_prob0.1.pth

        rm -r ${SYNFOLDER}

        let "idx+=1"

    else

        break

    fi

done
