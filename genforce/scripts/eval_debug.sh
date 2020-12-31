#!/bin/bash

LAMB=$1
METRIC=$2
BASELR=$3
N_IMAGES=$4
NETHZ=$5



for idx in {1..9};

do

BAR="_"
SAVENAME="${LAMB}$BAR${METRIC}$BAR${BASELR}"

FOLDER="/cluster/scratch/"
SLASH="/"
ENDING="_generator.pth"

echo "$FOLDER${NETHZ}$SLASH$SAVENAME$BAR$idx$ENDING"

done
