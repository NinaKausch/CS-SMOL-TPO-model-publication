import os
import pandas as pd
import numpy as np
import rdkit
import pdb

from chembl_structure_pipeline import standardizer
import Utils as utils
import argparse


def main(path, infile, sep, smiles_col, id_col, test_set, activity_col):
    absolute_path = os.path.dirname(__file__)
    if infile.endswith(".txt"):
        prefix = infile.split('.txt')[0]
    elif infile.endswith(".csv"):
        prefix = infile.split('.csv')[0]
    else:
        # Handle other cases or raise an error
        raise ValueError("Unsupported file extension")
    output = prefix + '_standardized.csv'

    df = pd.read_table(os.path.join(absolute_path, path, infile), sep=sep, decimal='.')
    print(df.shape)
    print(df.columns)

    # Standardize according to https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py;
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00456-1
    df['SMILES'] = df[smiles_col].apply(utils.chembl_standardize)

    # how many SMILES where transformed?
    diff_count = (df['SMILES'] != df[smiles_col]).sum()

    print(f"Number of cases where 'SMILES' is different from original input: {diff_count}")

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
                        help='does your file contain an Act_class-column? If yes, testset is true and you will have to specify activity_col')
    args = parser.parse_args()
    main(infile=args.infile, path=args.path, sep=args.sep, id_col=args.id_col, smiles_col=args.smiles_col,
         test_set=args.test_set, activity_col=args.activity_col)