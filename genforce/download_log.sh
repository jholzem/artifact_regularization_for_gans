#!/bin/bash


LAMBDAS=$1
METRIC=$2
LRS=$3
NETHZ=$4

LOCALFOLDER=$5


for LAMBDA in $LAMBDAS;

do

    for LR in $LRS;

    do

    BAR="_"
    SAVENAME="${LAMBDA}$BAR${METRIC}$BAR${LR}"

    FOLDER="@login.leonhard.ethz.ch:/cluster/scratch/"
    RES="/results/"
    TEMP="_workdir"
    LOG="/log"
    TXT=".txt"
    SLASH="/"

    FILE="${NETHZ}$FOLDER${NETHZ}$RES$SAVENAME$TEMP$LOG$TXT"

    scp $FILE ${LOCALFOLDER}

    mv "${LOCALFOLDER}$LOG$TXT" "${LOCALFOLDER}$SLASH$SAVENAME$TXT"

    done

done
