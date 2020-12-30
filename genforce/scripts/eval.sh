#!/bin/bash

N_IMAGES=$1
GENERATOR=$2

python img_syn.py ${N_IMAGES} 'styleganinv_ffhq256'

python demo_dir.py -d img_syn_genforce/styleganinv_ffhq256 -m weights/blur_jpg_prob0.1.pth
