from __future__ import print_function
import argparse
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

import os
from io import StringIO
import sys
import Utils as utils

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics._scorer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    precision_recall_curve, auc, make_scorer, recall_score, balanced_accuracy_score, accuracy_score, precision_score, \
    f1_score, matthews_corrcoef
import catboost as catboost
from catboost import CatBoostClassifier, Pool, cv

import shap
#######
# Input file should contain info in the following order, separated by semicolon:
# - ID-Column (i.e. BCS code) called 'ID'
# - SMILES column
# - Activity Column
# - Features generated with 01_datapreparation.py, 02_Make_RDKIT-features.py

# Example for running this script:

def main(path, input, model, cluster,tc):
    absolute_path = os.path.dirname(__file__)
    prefix = input.split('.csv')[0]

    df = pd.read_table(os.path.join(absolute_path, path, input), sep = ';', decimal = '.')

    print(df.shape)
    print(df.describe)

    # PHNS: remove nans
    df=df.dropna()  # remove compounds without RDkit descriptors

    X_train, X_test, y_train, y_test, X_id, df_feat, X_train_all = utils.split_train_test (df)

    # TODO: scale descriptors between 0 and 1?
    # e.g. Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data
    # PHNS

    X_id.to_csv(os.path.join(absolute_path, path, prefix+model+'_03_X_id.csv'))
    print('X_id.shape', X_id.shape)
    # save features expected by model:
    df_feat.head().to_csv(os.path.join(absolute_path, path, prefix+model+'_03_required_features.csv'), sep = ';', index = False)
    # Save Testset only for use in predict.py
    df_test = X_id[X_id['Train']==0]
    df_test = df_test.drop(columns=['Train'])
    df_test.to_csv(os.path.join(absolute_path, path, prefix + '_03_pure_test.csv'), sep = ';', index = False)

    if cv:
        if cluster:
            X_train_all = X_train_all.reset_index(drop=True)
            # TODO: add parameter for TC
            print('Sphere exclusion clustering started')
            df_with_assigned_clusters = utils.make_cluster_assignment(X_train_all, 'SMILES', tc)
            df_train_for_cluster = pd.merge(X_train_all, df_with_assigned_clusters, on='SMILES')
            df_train_for_cluster_print = df_train_for_cluster[['ID', 'SMILES', 'Cluster', 'Act_class']]

            df_train_for_cluster_print.to_csv(os.path.join(absolute_path, path, prefix + '_03_train_clustered_TC_'+str(tc)+'.csv'), sep = ';', index = False)

            groups_train_cluster = df_train_for_cluster['Cluster']
            y_train_cluster = df_train_for_cluster['Act_class']
            X_train_cluster = df_train_for_cluster.drop(columns=['ID', 'SMILES', 'Act_class', 'Cluster'])

            utils.tanimoto_cv(absolute_path, path, prefix, X_train_cluster, y_train_cluster, groups_train_cluster,tc )

        else:
            utils.model_cv(absolute_path, path, prefix, X_train, y_train)

    utils.train_final_model(absolute_path, path, prefix, model, X_train, X_test, y_train, y_test, X_id)

############################### MAIN ########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train_model')
    parser.add_argument('--path', action='store', default='data',
                        help='data path')
    parser.add_argument('--input', action='store', default='test_input.csv',
                        help='input data file, should contain ID, SMILES, Act_class in this order, seperated by semicolon; check for smile validity and no double smiles')

    parser.add_argument('--cv', action='store_true', default=False,
                        help='decide if you want to do 5-fold crossvalidation on X-train with default models lr, rf, gbt, catboost prior to training your model on full X_train')
    parser.add_argument('--model', action='store', default='catboost',
                        help='choose from catboost, gbt, rf')
    parser.add_argument('--cluster', action='store_true', default=False,
                        help='decide if you want to do random 5-fold cross-validation of assignment of folds based on chemical cluster')
    parser.add_argument('--TC', action='store', default=0.6, help='Tanimoto cut-off for clustering', type=float)

    args = parser.parse_args()
    main(input=args.input, path=args.path, model = args.model, cluster = args.cluster, tc=args.TC)

