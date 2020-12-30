#!/bin/bash

N_IMAGES=$1
GENERATOR=$2

python img_syn.py ${N_IMAGES} ${GENERATOR}

python demo_dir.py -d test_syn -m weights/blur_jpg_prob0.1.pth
