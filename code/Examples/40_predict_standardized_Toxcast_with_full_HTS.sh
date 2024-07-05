#!/bin/bash

source ~/.bashrc

conda activate base


python /home/gfuis/CS-SMOL-TPO_model_publication/code/04_Predict.py \
       --path data \
       --input TPO_Toxcast_BASF_standardized_02_train_test.csv \
       --trained_model TPO_full_HTS_standardized_02_train_testcatboost.cbm \
       --features TPO_full_HTS_standardized_02_train_testcatboost_03_required_features.csv \
       --test_set \

	
