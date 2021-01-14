CONF_F=$1
EP_F=$2
CONF_FA=$3
EP_FA=$4
CONF_A=$5
EP_A=$6
NETHZ=$7

FOLDER="/cluster/scratch/"
RES="/results/"

RESFOLDER="$FOLDER${NETHZ}$RES"

python visualize.py ${CONF_F} ${EP_F} ${CONF_FA} ${EP_FA} ${CONF_A} ${EP_A} ${RESFOLDER}
