#!/bin/bash

python img_syn.py 10 'styleganinv_ffhq256'

python demo_dir.py -d img_syn_genforce/$1 -m weights/blur_jpg_prob0.1.pth
