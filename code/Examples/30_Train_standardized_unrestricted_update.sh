#!/bin/bash

source ~/.bashrc

conda activate base

python /home/gfuis/CS-SMOL-TPO_model_publication/code/03_Train_model.py \
      --path data \
      --input S2_Result_list_updated_standardized_02_train_test.csv \
      --model catboost \
      --cv \
      --cluster

