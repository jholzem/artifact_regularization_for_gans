#!/bin/bash

##################################

FOLDER="workdir/accuracies/*"
idx=1
for file in $FOLDER;
do

  acc[$idx]=$file
  idx=$((idx + 1))

done
python plot_acc.py "gen_visualizations/" "${acc[1]}" "${acc[2]}" "${acc[3]}"

##################################

FOLDER="workdir/advLoss/*"
idx=1
for file in $FOLDER;
do
  adv[$idx]=$file
  idx=$((idx + 1))

done
python plot_aL.py "gen_visualizations/" "${adv[1]}" "${adv[2]}" "${adv[3]}"

##################################

FOLDER="workdir/fourierLoss/*"
idx=1
for file in $FOLDER;
do
  fou[$idx]=$file
  idx=$((idx + 1))

done
python plot_fL.py "gen_visualizations/" "${fou[1]}" "${fou[2]}" "${fou[3]}"

##################################

FOLDER="workdir/generators/*"
for file in $FOLDER;
do

  python img_syn_fromLatent.py "$file" "gen_visualizations/images426" "workdir/latentCodes" "00426.csv"
  python img_syn_fromLatent.py "$file" "gen_visualizations/images388" "workdir/latentCodes" "00388.csv"
  python img_syn_fromLatent.py "$file" "gen_visualizations/images716" "workdir/latentCodes" "00716.csv"

done

