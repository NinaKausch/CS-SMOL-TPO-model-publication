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
import scipy.stats
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
import os
from io import StringIO
import sys

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import collections
from collections import defaultdict
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import pandas as pd
import pickle
import scipy.stats as stats
import numpy as np
import Utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from dataclasses import dataclass
from typing import Optional, Any, List, Dict
import sys

#######
# Input file should contain info in the following order, separated by tab:
# - ID-Column (i.e. BCS code) called 'ID'
# - SMILES column
# - Activity Column called "Activity", default is that activity is already a classifier like Act_class (if this is a train set)

# Example for running this script: python CS-SMOL-TECHFU-TPO_model/code/02_Make_RDKIT_features.py --path data --input ToxCast21_Smiles_TPO_act.csv --Train

# if Test_train = True: A Test/Training set with existing TPO-data is provided; Enrichment protocol with identification of bits most strongly associated with TPO activity
# output: df with all features from rd_kit, Enrichment; csv_file with list of columns (i.e. features which are needed for model)

# if Test_train = False: Calculation of fixed bits handed over from csv_file


def main(path, infile, sep, enrich, Train, bits_required_by_model, TPO_act_tresh, p_value, sub_bit_count, Bayer_Models, Bayer_model_file):
    absolute_path = os.path.dirname(__file__)
    prefix = infile.split('.csv')[0]
    bayer_models = prefix + '_02_merged.csv'
    output_bits = prefix + '_02_bits.csv'
    output_enriched_bits = prefix + '_02_enriched_bits.csv'
    train_test_set = prefix + '_02_train_test.csv'

    suppl = Chem.SmilesMolSupplier(os.path.join(absolute_path, path, infile), delimiter = '\t', smilesColumn = 1, nameColumn = 0)
    data = pd.read_table(os.path.join(absolute_path, path, infile), sep = '\t', decimal = '.')
    print('shape of original dataset', data.shape)

    print(data.head())
    data['ID'] = data['ID'].astype(str)

    data_ind = data.set_index('ID')
    cmpd_list = list(data_ind.index)
    smiles = dict(zip(data_ind.index.values, data_ind.SMILES))
    (molecules, BitInfo, BitCount, BitCmpd)  = utils.molSupptoFpBits(suppl, cmpd_list, r = 3)
    data['BitInfo'] = data['ID'].map(BitInfo)

    # Calculate RDKIT-Properties for SMILES
    data = utils.getRDKitDescriptors(data, smiles_col='SMILES')
    print('shape of df:', data.shape)
    print(data.head())
    data = data.dropna(subset=['SMILES', 'mol'], axis=0)
    data = data.mask(data.eq('None')).dropna()
    print('shape of df: na SMILES, mol (& IDs) removed', data.shape)

    data.to_csv(os.path.join(absolute_path, path, output_bits), index=False, sep=';')

# In Case of Train: calculate which bits are enriched in TPO-active compounds; create 'Act_class column for training classifier
    if enrich:
        if Train:
            data['Act_class'] = data['Activity'].apply(lambda x: 1 if x >= float(TPO_act_tresh) else 0)

            data_ind = data_ind.loc[molecules.keys(), :]
            data_ind = data_ind.loc[molecules.keys(), :]
            data_ind = data_ind[(data_ind['Activity'] >= float(TPO_act_tresh))]

            ID_list = list(data_ind[(data_ind['Activity'] >= float(TPO_act_tresh))].index)
            print("Number of samples after filtering: " + str(len(ID_list)))
            (SubBitCount, SubBitCmpd) = utils.subsetBitCount(BitInfo, ID_list)

            E_ScoreSubsetBit = dict()
            for bit in SubBitCount:
                E_ScoreSubsetBit[bit] = (SubBitCount[bit] / len(ID_list)) / (BitCount[bit] / len(suppl))

            results_i = pd.DataFrame(
                columns=["Bit", "Bit_count_TPO_active", "Enrichment", "Bit_count_all", "oddsRatio", "pvalue", "Smiles",
                     "ROMol"])

            i = 0
            for k in sorted(E_ScoreSubsetBit, key=lambda k: E_ScoreSubsetBit[k], reverse=True):

                if (SubBitCount[k] > int(sub_bit_count)) & (len(molecules.keys()) > BitCount[k]):
                    oddsratio, pvalue = stats.fisher_exact(
                        utils.contingency_table(len(molecules.keys()), len(ID_list), BitCount[k], SubBitCount[k]))

                    if pvalue <= float(p_value):
                        m = SubBitCmpd[k][0]
                        aid, rad = BitInfo[m][k][0]
                        smarts = utils.match_substr(molecules[m], aid, rad)
                        results = pd.DataFrame(
                            columns=["Bit", "Bit_count_TPO_active", "Enrichment", "Bit_count_all", "oddsRatio", "pvalue",
                                 "Smiles", "BitCmpd"])
                        results.loc[i] = [str(k), SubBitCount[k], E_ScoreSubsetBit[k], BitCount[k], oddsratio, pvalue, smiles[m], BitCmpd[k]]

                        PandasTools.AddMoleculeColumnToFrame(results)
                        results[results.ROMol >= rdkit.Chem.MolFromSmarts(smarts)]
                        results_i = results_i.append(results, ignore_index=True)
                        i += 1
            results_i['Bit_proportion'] = results_i.apply(lambda x: x["Bit_count_all"] / (len(molecules.keys())), axis=1)
            results_f = results_i[results_i['Bit_proportion'] < 0.25]
            # Results_f contains a list of all bits that are enriched in TPO-active compounds
            results_f.to_csv(os.path.join(absolute_path, path, output_enriched_bits), index=False, sep=';')

        data_bits = pd.read_table(os.path.join(absolute_path, path, output_bits), sep = ';', decimal = '.')

        if Train:
           results_f = pd.read_table(os.path.join(absolute_path, path, output_enriched_bits), sep = ';', decimal = '.')
           print('Use Train', results_f.shape)
        else:
            results_f = pd.read_table(os.path.join(absolute_path, path, bits_required_by_model), sep=';', decimal='.')
            print('Use bits from trained model', results_f.shape)
            if 'Activity' in data_bits.columns:
               data_bits['Act_class'] = data['Activity'].apply(lambda x: 1 if x >= float(TPO_act_tresh) else 0)

        if Train:
            data_bits['Act_class'] = data_bits['Activity'].apply(lambda x: 1 if x >= float(TPO_act_tresh) else 0)
            plt.rcParams["figure.figsize"] = (3, 3)
            ax = sns.histplot(data = data_bits, x="Activity", hue = 'Act_class')
            plt.savefig(os.path.join(absolute_path, path, prefix + 'Hist_Activity_Act_class.png'))
            hitrate = (sum(data_bits['Act_class'])*100 / data_bits.shape[0])
            print('number of samples', data_bits.shape[0])
            print('hitrate_', hitrate )

        # remove cpds without bit info
        print('original df shape: ', data_bits.shape)
        data_bits.dropna(subset=['BitInfo'], inplace=True)
        print('df shape after removing rows without bit_info: ', data_bits.shape)

        # All bits per cpd:
        Bit_list = list()
        for i in data_bits['BitInfo']:
            Bit_list_i = list(ast.literal_eval(str(i)).keys())
            Bit_list.append(Bit_list_i)

        data_bits['Bit_list'] = Bit_list

        # List of enriched bits per Substance
        data_bits['enriched'] = data_bits['Bit_list'].apply(lambda x: sorted([i for i in x if i in results_f['Bit'].values]))

        print(data_bits[0:15])

        # one-hot-encoding
        fbits_encoded = pd.get_dummies(data_bits.enriched.apply(pd.Series).stack()).sum(level=0)

        print(fbits_encoded[0:15])

        # merge encoded bits with data
        df_encoded = data_bits.merge(fbits_encoded, left_index=True, right_index=True, how = 'left')
        df_encoded.columns = df_encoded.columns.map(str)
        print('shape of df', df_encoded.shape)
        df_encoded.drop(columns=['Bit_list', 'enriched'], axis=1, inplace=True)

    else:
        if Train:
            data['Act_class'] = data['Activity'].apply(lambda x: 1 if x >= float(TPO_act_tresh) else 0)
            hitrate = (sum(data['Act_class']) * 100 / data.shape[0])
            print('number of samples', data.shape[0])
            print('hitrate_', hitrate)
        # compute ECFP4 instead of enriched fingerprints
        feature = pd.DataFrame(data['SMILES'].apply(lambda x: utils.to_fp(x, fingerprint_bits=1024, fingerprint_radius=3)).to_numpy().tolist())
        feature.columns = feature.columns.map(str)
        df_ecfp4 = pd.concat([data, feature], axis=1)
        df_encoded = df_ecfp4

    # remove rows without valid mol, duplicated Smiles, IDs etc
    df_encoded = df_encoded.drop_duplicates(subset=['ID', 'SMILES'])
    print('shape of df: duplicate ID, SMILES removed', df_encoded.shape)
    #df_encoded = df_encoded.dropna(subset=['ID', 'SMILES', 'mol'], axis=0)
    #print('shape of df: na SMILES, mol & IDs removed', df_encoded.shape)
    df_encoded.drop(columns=['BitInfo', 'mol', 'Activity'], axis=1, inplace=True)
    print('df shape after merging with one hot enc bits, removal of empty mol, duplicated IDs: ', df_encoded.shape)

    #for col in df_encoded.columns:
    #    pct_missing = np.mean(df_encoded[col].isnull())
    #    print('{} - {}%'.format(col, round(pct_missing * 100)))


    #df_encoded.dropna(axis=0, how='any', inplace=True)
    #print('df after removing any na rows', df_encoded.shape)

    # If Bayermodels are used they are added in this step to the final dataframe
    if Bayer_Models:
        df_bayer_models = pd.read_table(os.path.join(absolute_path, path, Bayer_model_file), sep=';', decimal='.')
        print('Bayer_model_output_shape', df_bayer_models.shape)
        df_encoded = pd.merge(df_encoded, df_bayer_models, left_on = 'SMILES', right_on = 'col1', how = 'left')
        df_encoded.dropna(axis=0, how='any', inplace=True)
        print('df shape after merging RDKIT with Bayermodel_output and removing nans: ', df_encoded.shape)
        df_encoded = df_encoded.drop(columns=['col1', 'smiles_used'])

        hitrate = (sum(df_encoded['Act_class']) * 100 / df_encoded.shape[0])
        print('number of samples', df_encoded.shape[0])
        print('hitrate_', hitrate)

    df_encoded.to_csv(os.path.join(absolute_path, path, train_test_set), index=False, sep=';')


############################### MAIN ########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='make_bits')
    parser.add_argument('--path', action='store', default='data',
                        help='data path')
    parser.add_argument('--input', action='store', default='test_input.csv',
                        help='input data file, should contain ID, SMILES, Activity (optional) in this order, seperated by semicolon; check for smile validity and no double smiles')
    parser.add_argument('--enrich', action='store_true', default=False,
                        help='if enrich is True, statistically enriched fingerprints are calculated - this makes only sense with large datasets (> 2000 cpds). If Enrich is not set to true ECFP4 are calculated')
    parser.add_argument('--Train', action ='store_true', default = False,
                        help = 'if Train is True, provide column with TPO-Activity')
    parser.add_argument('--sep', action='store', default=';',
                        help='specify seperator of infile')
    parser.add_argument('--Threshold', action ='store', default = 1,
                        help='define minimum threshold for TPO activity, everything >= threshold is considered active; Column name should be "Activity"')
    parser.add_argument('--p_value', action='store', default=0.001,
                        help='define p_value for enrichment')
    parser.add_argument('--sub_bit_count', action='store', default=100,
                        help='define minimum how often a bit has to occur in the dataset to be considred for enrichment; 100 suggested for dataset > 10K, 25 for smaller datasets')
    parser.add_argument('--bits_required_by_model', action='store', default='ToxCast21_Smiles_TPO_act_clean_02_enriched_bits.csv',
                        help='If you want to use a model that has already been trained - i.e. if Train is False - provide the _enriched_bits file that contains the expected bits')
    parser.add_argument('--Bayer_Models', action='store_true', default=False,
                        help='if Bayer_Models is True, csv_file containing Bayer internal features from 01_datapreparation.py is merged on SMILES column; fileformat should be prefix + _merged.csv')
    parser.add_argument('--Bayer_model_file', action='store', default='data',
                        help='data path')
    args = parser.parse_args()
    main(infile=args.input, path=args.path, enrich=args.enrich, Train=args.Train, sep = args.sep, TPO_act_tresh = args.Threshold, p_value = args.p_value, sub_bit_count = args.sub_bit_count,
         bits_required_by_model = args.bits_required_by_model, Bayer_Models=args.Bayer_Models, Bayer_model_file = args.Bayer_model_file)
