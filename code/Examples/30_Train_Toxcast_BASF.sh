#!/bin/bash

source ~/.bashrc

conda activate base

python /home/gfuis/CS-SMOL-TECHFU-TPO_model/code/03_Train_model.py \
      --path data \
      --input TPO_Toxcast_BASF_02_train_test.csv \
      --model catboost \
      --cv \
      --cluster \
      --TC 0.2