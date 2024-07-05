#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import sys
import ast
from dataclasses import dataclass
from typing import Optional, Any, List, Dict
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D
from rdkit.Chem.Draw import IPythonConsole
import collections
from collections import defaultdict
import pickle
import Utils as utils


#######
# Input file should contain info in the following order, separated by tab:
# - ID-Column (i.e. BCS code) called 'ID'
# - SMILES column
# - Activity Column called "Activity", default is that activity is already a classifier like Act_class (if this is a train set)

# Example for running this script: python CS-SMOL-TECHFU-TPO_model/code/02_Make_RDKIT_features.py --path data --input ToxCast21_Smiles_TPO_act.csv --Train

# if Test_train = True: A Test/Training set with existing TPO-data is provided; Enrichment protocol with identification of bits most strongly associated with TPO activity
# output: df with all features from rd_kit, Enrichment; csv_file with list of columns (i.e. features which are needed for model)

# if Test_train = False: Calculation of fixed bits handed over from csv_file


def main(path, input, sep, Train, TPO_act_tresh, Bayer_Models, Bayer_model_file):
    absolute_path = os.path.dirname(__file__)
    prefix = input.split('.csv')[0]
    bayer_models = prefix + '_02_merged.csv'
    train_test_set = prefix + '_02_train_test.csv'

    df = pd.read_table(os.path.join(absolute_path, path, input), sep = '\t', decimal = '.')
    print('shape of df:', df.shape)

    # Calculate RDKIT-Properties for SMILES
    df = utils.getRDKitDescriptors(df, smiles_col='SMILES')
    print(df.head())
    df = df.dropna(subset=['ID', 'SMILES'], axis=0)
    df = df.mask(df.eq('None')).dropna()
    df = df.drop_duplicates(subset=['ID', 'SMILES'])
    print('shape of df: rdkit descriptors added, na SMILES, mol & IDs, duplicates removed', df.shape)

    # calculate Morgan Fingerprints
    df_morgan = df.copy()
    df_morgan = utils.build_dataframe(df_morgan)
    morgan_df = pd.DataFrame(df_morgan['morgan'].tolist(), index=df_morgan.index)
    morgan_df.columns = [i for i in range(morgan_df.shape[1])]
    df_expanded = pd.concat([df_morgan, morgan_df], axis=1)
    df_expanded.drop(columns = ['morgan','mol'], inplace=True)
    df = df.merge(df_expanded, on = 'SMILES')
    print('shape of df after adding fp', df.shape)

    if Train:
        df['Act_class'] = df['Activity'].apply(lambda x: 1 if x >= float(TPO_act_tresh) else 0)
        hitrate = (sum(df['Act_class']) * 100 / df.shape[0])
        print('number of samples', df.shape[0])
        print('hitrate_', hitrate)
        df = df.drop(columns=['Activity'])

    # remove rows without valid mol, duplicated Smiles, IDs etc
    df = df.dropna(subset=['ID', 'SMILES', 'mol'], axis=0)
    df.drop(columns=['mol'], inplace=True)
    print('shape of df: na ID, SMILES, mol removed', df.shape)
    print('df.columns', list(df.columns))


    # If Bayermodels are used they are added in this step to the final dataframe
    if Bayer_Models:
        df_bayer_models = pd.read_table(os.path.join(absolute_path, path, Bayer_model_file), sep=';', decimal='.')
        print('Bayer_model_output_shape', df_bayer_models.shape)
        df = pd.merge(df, df_bayer_models, left_on = 'SMILES', right_on = 'col1', how = 'left')
        df.dropna(axis=0, how='any', inplace=True)
        print('df shape after merging RDKIT with Bayermodel_output and removing nans: ', df.shape)
        df = df.drop(columns=['col1', 'smiles_used'])

        hitrate = (sum(df['Act_class']) * 100 / df.shape[0])
        print('number of samples', df.shape[0])
        print('hitrate_', hitrate)

    df.to_csv(os.path.join(absolute_path, path, train_test_set), index=False, sep=';')


############################### MAIN ########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='make_features')
    parser.add_argument('--path', action='store', default='data',
                        help='data path')
    parser.add_argument('--input', action='store', default='test_input.csv',
                        help='input data file, should contain ID, SMILES, Activity (optional) in this order, seperated by semicolon; check for smile validity and no double smiles')
    #parser.add_argument('--enrich', action='store_true', default=False,
    #                    help='if enrich is True, statistically enriched fingerprints are calculated - this makes only sense with large datasets (> 2000 cpds). If Enrich is not set to true ECFP4 are calculated')
    parser.add_argument('--Train', action ='store_true', default = False,
                        help = 'if Train is True, provide column with TPO-Activity')
    parser.add_argument('--sep', action='store', default=';',
                        help='specify seperator of infile')
    parser.add_argument('--Threshold', action ='store', default = 1,
                        help='define minimum threshold for TPO activity, everything >= threshold is considered active; Column name should be "Activity"')
    #parser.add_argument('--p_value', action='store', default=0.001,
    #                    help='define p_value for enrichment')
    #parser.add_argument('--sub_bit_count', action='store', default=100,
    #                    help='define minimum how often a bit has to occur in the dataset to be considred for enrichment; 100 suggested for dataset > 10K, 25 for smaller datasets')
    #parser.add_argument('--bits_required_by_model', action='store', default='ToxCast21_Smiles_TPO_act_clean_02_enriched_bits.csv',
    #                    help='If you want to use a model that has already been trained - i.e. if Train is False - provide the _enriched_bits file that contains the expected bits')
    parser.add_argument('--Bayer_Models', action='store_true', default=False,
                        help='if Bayer_Models is True, csv_file containing Bayer internal features from 01_datapreparation.py is merged on SMILES column; fileformat should be prefix + _merged.csv')
    parser.add_argument('--Bayer_model_file', action='store', default='data',
                        help='data path')
    args = parser.parse_args()
    main(input=args.input, path=args.path, Train=args.Train, sep = args.sep, TPO_act_tresh = args.Threshold, Bayer_Models=args.Bayer_Models, Bayer_model_file = args.Bayer_model_file)
