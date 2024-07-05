#!/bin/bash

source ~/.bashrc

conda activate base


python /home/gfuis/CS-SMOL-TPO_model_publication/code/04_Predict.py \
       --path data \
       --input TPO_new_holdout_2021_standardized_02_train_test.csv \
       --trained_model S2_Result_list_updated_standardized_02_train_testcatboost.cbm \
       --features S2_Result_list_updated_standardized_02_train_testcatboost_03_required_features.csv \
       --test_set \

	
