ADV=$1
LAMB=$2
METRIC=$3
BASELR=$4
N_IMAGES=$5
NETHZ=$6
PORT=$RANDOM

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
           --adv=${ADV} \
           --lamb=${LAMB} \
           --metric=${METRIC} \
           --baseLR=${BASELR} \
           --nethz=${NETHZ}
	   ${@:8}


idx=1

while true;

do

    FILE="$FOLDER${NETHZ}$RES$SAVENAME$BAR$idx$ENDING"

    if test -f "$FILE"; then

        python img_syn.py ${N_IMAGES} ${FILE} ${SYNFOLDER}
        python CNNDetection/detection.py -d ${SYNFOLDER}

        rm -r ${SYNFOLDER}

        let "idx+=1"

    else

        break

    fi

done


