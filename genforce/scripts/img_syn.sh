#!/bin/bash

N_IMAGES=$1
GENERATOR=$2
SYNDIR=$3

python img_syn.py ${N_IMAGES} ${GENERATOR} ${SYNDIR}

python demo_dir.py -d ${SYNDIR} -m weights/blur_jpg_prob0.1.pth

rm -r ${SYNDIR}
