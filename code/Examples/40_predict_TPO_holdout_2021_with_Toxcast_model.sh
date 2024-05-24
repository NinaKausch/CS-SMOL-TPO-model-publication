#!/bin/bash

source ~/.bashrc

conda activate base


python /home/gfuis/CS-SMOL-TECHFU-TPO_model/code/04_Predict.py \
       --path  data \
       --input  TPO_2021_holdout_orig_feat.csv\
       --trained_model publish_dataset_02_train_testcatboost.cbm \
       --features publish_dataset_02_train_testcatboost_03_required_features.csv \
       --test_set \

	
