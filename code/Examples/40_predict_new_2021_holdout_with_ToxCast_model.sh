#!/bin/bash

source ~/.bashrc

conda activate base


python /home/gfuis/CS-SMOL-TECHFU-TPO_model/code/04_Predict.py \
       --path  data \
       --input  TPO_new_Holdout_2021_02_train_test.csv \
       --trained_model TPO_Toxcast_BASF_02_train_testcatboost.cbm \
       --features TPO_Toxcast_BASF_02_train_testcatboost_03_required_features.csv \
       --test_set \

	
