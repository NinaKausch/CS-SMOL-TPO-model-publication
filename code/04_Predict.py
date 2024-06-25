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
# - Activity Column (optinal if this is a labeled Testset)
# - Features generated with 01_datapreparation.py, 02_Make_RDKIT-features.py
# please provide your trained model
# - please provide a file specifying the columns that the model expects
# run on command line like this: python CS-SMOL-TECHFU-TPO_model/code/04_Predict.py --path data --input .csv


def main(path, infile, trained_model, features, test_set):
    absolute_path = os.path.dirname(__file__)
    prefix1 = infile.split('.csv')[0]
    prefix2 = trained_model.split('.cbm')[0]

    df = pd.read_table(os.path.join(absolute_path, path, infile), sep = ';', decimal = '.')
    print(list(df['Act_class']))

    print(df.shape)
    print(df.describe)
    print(list(df.columns))

    # Load trained model from in-memory file object
    from_file = CatBoostClassifier()
    model = from_file.load_model(os.path.join(absolute_path, path, trained_model))

    # load features and adjust df
    X = df
    df_feat = pd.read_table(os.path.join(absolute_path, path, features), sep = ';', decimal = '.')
    missing_features = df_feat.columns.difference(X.columns).tolist()
    print('missing_features_',missing_features)
    X[missing_features] = 0
    redundant_features = X.columns.difference(df_feat.columns).tolist()
    print('redundant_features_', redundant_features)
    X = X.drop(columns = redundant_features)
    X = X[df_feat.columns]


    if test_set:
        #id_list = ['Act_class', 'ID', 'SMILES']
        #X = df.drop(columns = [id_list])
        y = df['Act_class']
        y_hat = model.predict(X)
        y_proba = model.predict_proba(X)
        y_hat = pd.DataFrame(y_hat)
        y_proba = pd.DataFrame(y_proba)
        df['y_hat'] = y_hat
        df['proba'] = y_proba[1]
        #df['proba_inactive'] = y_proba[:, 0]

        utils.test_vs_train_scores(absolute_path, path, prefix1, model, df, y_hat, y_test=y, X_test=X)

        df.to_csv(os.path.join(absolute_path, path, prefix1 + prefix2 +'_04_testset_predicted.csv'))

    else:
        y_hat = model.predict(X)
        y_proba = model.predict_proba(X)
        y_hat = pd.DataFrame(y_hat)
        y_proba = pd.DataFrame(y_proba)
        df['y_hat'] = y_hat
        #df['proba'] = y_proba[1]
        df['proba'] = y_proba[:, 1]
        df['proba_inactive'] = y_proba[:, 0]
        df.to_csv(os.path.join(absolute_path, path, prefix1, prefix2 + '_04_predicted.csv'))


############################### MAIN ########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Predict')
    parser.add_argument('--path', action='store', default='data',
                        help='data path')
    parser.add_argument('--input', action='store', default='test_input.csv',
                        help='input data file, should contain ID, SMILES, Act_class (opt) in this order, seperated by semicolon; check for smile validity and no double smiles')
    parser.add_argument('--trained_model', action='store', default='HTS_TPO_origcatboost.cbm',
                        help='choose from catboost, gbt, rf')
    parser.add_argument('--features', action='store', default='HTS_TPO_orig_catboost_required_features.csv',
                        help='attach df that contains features for model')
    parser.add_argument('--test_set', action='store_true', default=False,
                        help='does your file contain an Act_class-column? If yes, testset is true')
    args = parser.parse_args()
    main(infile=args.input, path=args.path, trained_model = args.trained_model, features = args.features, test_set = args.test_set)