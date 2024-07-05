import os
import pandas as pd
import numpy as np
import rdkit
import pdb

from chembl_structure_pipeline import standardizer
import Utils as utils
import argparse


def main(path, infile, sep, smiles_col, id_col, get_parent, test_set, activity_col):
    absolute_path = os.path.dirname(__file__)
    if infile.endswith(".txt"):
        prefix = infile.split('.txt')[0]
    elif infile.endswith(".csv"):
        prefix = infile.split('.csv')[0]
    else:
        # Handle other cases or raise an error
        raise ValueError("Unsupported file extension")
    if get_parent:
        output = prefix + '_standardized.csv'
    else:
        output = prefix + '_standardized_only.csv'

    df = pd.read_table(os.path.join(absolute_path, path, infile), sep=sep, decimal='.')
    print(df.shape)
    print(df.columns)

    # Standardize according to https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py;
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00456-1
    df['std_SMILES'] = df[smiles_col].apply(utils.standardize_chembl)
    df['parent_SMILES'] = df['std_SMILES'].apply(utils.get_parent_chembl)

    # how many SMILES where transformed?
    diff_count = (df['std_SMILES'] != df[smiles_col]).sum()
    print(f"Number of cases where 'std_SMILES' is different from original input: {diff_count}")
    diff_count = (df['std_SMILES'] != df['parent_SMILES']).sum()
    print(f"Number of cases where 'parent_SMILES' is different from std_SMILES: {diff_count}")
    if get_parent:
        df = df.drop(columns = [smiles_col, 'std_SMILES'])
        df = df.rename(columns={'parent_SMILES':'SMILES'})
        print('get parent')
    else:
        df = df.drop(columns=[smiles_col, 'parent_SMILES'])
        df = df.rename(columns={'std_SMILES': 'SMILES'})
        print('only standardized')

    # remove duplicate SMILES
    df = df.drop_duplicates(subset=['SMILES'])

    if test_set:
        df = df[[id_col, 'SMILES', activity_col]]
        df = df.rename(columns={activity_col: 'Activity', id_col: 'ID'})
        #df = df[df['Activity'] != 2]
        print('df[Activity].unique()', df['Activity'].unique())
        df['Activity'] = df['Activity'].replace(2, 0)
        print('df[Activity].unique()', df['Activity'].unique())
    else:
        df = df[[id_col, 'SMILES']]
        df = df.rename(columns={id_col: 'ID'})

    print(df.shape)
    df = df.dropna()
    print(df.shape)
    print(df.columns)

    df.to_csv(os.path.join(absolute_path, path, output), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='standardize')
    parser.add_argument('--path', action='store', default='data', help='data path'),
    parser.add_argument('--infile', action='store', default='test_input.txt',
                        help='input data file, should contain id-col, smiles-col, Act_class (opt), seperated by sep'),
    parser.add_argument('--sep', action='store', default='\t', help='seperator of your file'),
    parser.add_argument('--id_col', action='store', default='id', help='unique identifier for your SMILES'),
    parser.add_argument('--smiles_col', action='store', default='Structure (as SMILES)',
                        help='name of your smiles col, beware: the new, standardized column will just be named SMILES'),
    parser.add_argument('--activity_col', action='store', default='activity class',
                        help='in case you have measured activities, select the column you want to use for activity classification'),
    parser.add_argument('--test_set', action='store_true', default=False,
                        help='does your file contain an Act_class-column? If yes, testset is true and you will have to specify activity_col'),
    parser.add_argument('--get_parent', action='store_true', default=False,
                        help='do you want to desalt and neutralize?')
    args = parser.parse_args()
    main(infile=args.infile, path=args.path, sep=args.sep, id_col=args.id_col, smiles_col=args.smiles_col,
         test_set=args.test_set, activity_col=args.activity_col, get_parent=args.get_parent)