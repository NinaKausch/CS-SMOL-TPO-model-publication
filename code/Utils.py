import subprocess
import os
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import collections
from collections import defaultdict
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D
from rdkit.Chem import Draw
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.DataManip.Metric import GetTanimotoSimMat

import subprocess
import sys
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from io import StringIO
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

######################################## RUN INTERNAL PHYSCHEM MODELS ##########################################################################


def calculate_physchem(infile, output_physchem, use_smiles=True):

    cmd_str = 'python /storage/home/projects/seedgrowthtf/PhyschemPi50/PhyschemPi50.py -infile ' + \
              infile + ' -outfile ' + output_physchem + ' -append_to_input -store_password -ignore_pi50'
    if use_smiles:
        cmd_str += ' -use_smiles'
    subprocess.call(cmd_str, shell=True)

def calculate_chargestate(physchem_df, input_chargestate, output_chargestate):
    if '/' in input_chargestate:
        infile = input_chargestate.split('/')[-1]
        data_path = input_chargestate.replace(infile, '')
    else:
        infile = input_chargestate
        data_path = '' # avoid TypeError when concatenating cmd_str
    if '/' in output_chargestate:
        outfile = output_chargestate.split('/')[-1]
    else:
        outfile = output_chargestate

    smi_df = physchem_df.dropna(subset=['smiles_used'])
    smi_df[['smiles_used', 'col1']].to_csv(input_chargestate, header=None, sep='\t', index=False)

    cmd_str = 'python ' + os.path.dirname(os.path.abspath
        (__file__)) + '/ChargeStateWrapper.py --infile ' + infile + ' --outfile ' + outfile + ' --path ' +  os.path.dirname \
        (os.path.abspath(__file__))
    if data_path != '':
        cmd_str += ' --data_path ' + data_path
    subprocess.call(cmd_str, shell=True)

#################################################### ChemBL SMILES CLEANING #################################################################

from chembl_structure_pipeline import standardizer

def standardize_chembl(smiles):
    '''Function for default standardization as employed in ChEMBl. Copyright (c) 2019 Greg Landrum;
    see: https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py
         https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00456-1'''

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Convert the Mol object to a molblock string
        molblock = Chem.MolToMolBlock(mol)
        # Now, pass the molblock string to standardize_molblock
        std_molblock = standardizer.standardize_molblock(molblock)
        # Convert the standardized molblock back to a Mol object
        std_mol = Chem.MolFromMolBlock(std_molblock)
        # Convert the Mol object to SMILES
        return Chem.MolToSmiles(std_mol)
    else:
        return None


def get_parent_chembl(smiles):
    '''Applies default tranformation to a SMILES string and returns the parent molecule SMILES string. neutralize=True, check_exclusion=True'''

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        result = standardizer.get_parent_mol(mol)

        # If a tuple is returned, extract the ROMol object; adjust this based on the actual structure of the return value
        if isinstance(result, tuple):
            parent_mol = result[0]  # Assuming the ROMol object is the first element of the tuple
        else:
            parent_mol = result

        # Ensure parent_mol is an RDKit ROMol object before converting to SMILES
        if isinstance(parent_mol, Chem.rdchem.Mol):
            return Chem.MolToSmiles(parent_mol)
        else:
            raise TypeError("Expected RDKit ROMol object, got something else.")
    else:
        return None
#################################################### BASIC SMILES CLEANING #################################################################

def standardized_mol(smiles):
    # SOURCE: https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    taut_uncharged_parent_clean_mol = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
            clean_mol = rdMolStandardize.Cleanup(mol)
            # if many fragments, get the "parent" (the actual mol we are interested in)
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            # try to neutralize molecule
            uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
            uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            te = rdMolStandardize.TautomerEnumerator()  # idem
            taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        if mol is None:
            taut_uncharged_parent_clean_mol = 'remove'
    finally:
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim is to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        return taut_uncharged_parent_clean_mol


##################################################### BASIC MOLFILE CREATION #######################################################################

#This is the simple alternative to standardized_mol, which is now used in the Calculate RDKIT section.

def make_mol(row, silent=True):
    # redirect stderr to silence rdkit
    if silent:
        io_orig = sys.stderr
        io = sys.stderr = StringIO()
    try:
        res = Chem.MolFromSmiles(row)
    except:
        res = None
    # restore stderr
    if silent:
        sys.stderr = io_orig
    return res

##################################################### CALCULATE RDKIT #######################################################################

def getRDKitDescriptors(df, mol_col='mol', smiles_col='SMILES'):
    if mol_col not in df.columns:
        df[mol_col] = df[smiles_col].apply(make_mol)
        #df = df[df[mol_col]!= 'remove']
    for descr in Descriptors._descList:
        df[descr[0]] = df[mol_col].apply(lambda x: x if x is None else descr[1](x))
    return df

#####################################  CALCULATE ENRICHTED MORGAN FINGERPRINTS ###########################################################


def molSupptoFpBits(molsupplier, compounds, r=3, bitsize=1024):
    """
    Input

    Function input is molSupplier object with mol name and SMILES string, plus two optional parameters:
    r - radius of Morgan fingerprint; bitsize - number of bits used for Morgan fingerprints


    Returns

    molDep - a dictionary where key is a BCS-code and value is mol object

    FpBitInfo - a dictionary where key is a BCS-code (molecule name) and value is another dictionary
    with key equal to Morgan bit and value bit info

    FpCMpd - a dictionary where key is fingerprint bit and value count how often a given bit occurs in a data set

    """

    from rdkit.Chem import AllChem
    from collections import Counter
    from collections import defaultdict

    FpBitInfo = dict()
    FpCmpdCount = Counter()
    FpCmpd = defaultdict(list)
    molDep = dict()
    i = 0

    print(r)

    for mol in molsupplier:
        if mol is None: continue  # write so that it warns which molecules were not defined
        info = dict()
        if i % 10000 == 0: print(i)  # nur jeder 10.000 te Schritt wird geprinted
        i += 1
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = r ,nBits = bitsize, bitInfo = info)
        fp = AllChem.GetMorganFingerprint(mol, radius=r,
                                          bitInfo=info)  # Fehlt bitsize versehentlich oder ist das Absicht?
        FpBitInfo[mol.GetProp("_Name")] = dict(info)
        molDep[mol.GetProp("_Name")] = mol
        # for bit in info.keys():
        for bit in fp.GetNonzeroElements():
            FpCmpdCount[bit] += 1
            FpCmpd[bit].append(mol.GetProp("_Name"))

    return (molDep, FpBitInfo, FpCmpdCount, FpCmpd)


def subsetBitCount(bit_info, cmpd_list):
    """ Returns a dictionary with bits count for data subset (cmpd_list)"""

    from collections import Counter
    from collections import defaultdict

    subset_fp_count = Counter()
    subset_list = defaultdict(list)

    for cmpd in cmpd_list:
        for fingerprint in bit_info[cmpd]:
            subset_fp_count[fingerprint] += 1
            subset_list[fingerprint].append(cmpd)

    return (subset_fp_count, subset_list)


def contingency_table(ds, sub_ds, bit, bit_sub):
    """

    Builds contingency table for Fisher exact test

    Input

    ds - number of unique compounds in a whole data set
    sub_ds - number of unique compounds in a subset of the data set

    bit - count of bit in whole set
    bit_sub - count of bit in a subset


    """

    bit_NOT_sub_ds = bit - bit_sub
    NO_bit_sub_ds = sub_ds - bit_sub
    NO_bit_NOT_sub_ds = ds - sub_ds - bit_NOT_sub_ds

    table = [[bit_sub, bit_NOT_sub_ds], [NO_bit_sub_ds, NO_bit_NOT_sub_ds]]

    return (table)


def match_substr(m, a, b):
    """
    Outputs SMARTS from a bit of a fingerprint

    m - mol object
    a - central atom in fingerprint bit
    b - bond length in a fingerprint bit

    """

    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import AllChem
    from rdkit.Chem import PandasTools
    from collections import defaultdict
    from rdkit.Chem import AllChem
    from collections import Counter
    from collections import defaultdict

    atoms = set()

    if b > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(m, b, a)
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())
    else:
        atoms = [a]
        env = None

    smarts = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms), bondsToUse=env, rootedAtAtom=a)

    return (smarts)

########################## Calculate ECFP4 #############################################################################

def to_fp(s: str, fingerprint_radius: int, fingerprint_bits: int):
    fp = None
    try:
        mol = make_mol(s)
        if mol is not None:
            bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=fingerprint_bits)
            print(type(bit_vector))
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(bit_vector, arr)
            fp = arr
    finally:
        return fp

###################### Calculate ECFP4 Version 2 #######################################################################
# See:
# https://practicalcheminformatics.blogspot.com/2020/03/benchmarking-one-molecular-fingerprint.html
# https://github.com/PatWalters/benchmark_map4/blob/master/benchmark_map4.ipynb

def fp_as_array(mol, fingerprint_radius, fingerprint_bits):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=fingerprint_bits)
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_dataframe(df):
    df = df[['SMILES']]
    df['mol'] = [Chem.MolFromSmiles(x) for x in df.SMILES]
    df['morgan'] = [fp_as_array(x, fingerprint_radius=3, fingerprint_bits=1024) for x in df.mol]
    return df

########################## Build Model #################################################################################

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics._scorer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    precision_recall_curve, auc, make_scorer, recall_score, balanced_accuracy_score, accuracy_score, precision_score, \
    f1_score, matthews_corrcoef
# deprecated: plot_confusion_matrix, multilabel_confusion_matrix, SCORERS,
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from lightgbm import LGBMClassifier
import catboost as catboost
from catboost import CatBoostClassifier, Pool, cv
import xgboost as xgb
from xgboost import XGBClassifier

import shap

def split_train_test (df):
    ''' annotate ID-columns, column to predict, feature columns; split into train and test set '''

    #df['Act_class'] = df['Activity'].apply(lambda x: 1 if x > float(threshold) else 0)
    #predict = ['Act_class']
    id_list = ['ID', 'SMILES', 'Act_class', 'Activity_percent']
    df_feat = df.drop(columns = ['Act_class', 'ID', 'SMILES', 'Activity_percent'], errors='ignore')
    #features = list(df_feat.columns)

    df.columns = df.columns.map(str)
    X_id = df
    y = df['Act_class']

    X_train_id, X_test_id, y_train, y_test = train_test_split(X_id, y, test_size=0.1, random_state=42, stratify=y)
    training_ids = list(X_train_id['ID'])
    X_id['Train'] = X_id['ID'].apply(lambda x: 1 if x in training_ids else 0)
    # PHNS
    X_train_all = X_train_id
    X_train = X_train_id.drop(columns=id_list, errors='ignore')
    print('X_train.columns', list(X_train.columns))
    X_test = X_test_id.drop(columns=id_list, errors='ignore')

    return X_train, X_test, y_train, y_test, X_id, df_feat, X_train_all

def tanimoto_cv(absolute_path, path, prefix, X_train, y_train, groups_train,tc):
    # '''stratified k-fold on X_train with chemical cluster always assigned to one fold, cv with lr, rf, gbt, catboost, make cv plots'''
    # return
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=50)
    gbt = GradientBoostingClassifier()
    svm_classifier = SVC(kernel='rbf',class_weight=class_weights)

    # lgbm = LGBMClassifier(scale_pos_weight=y.sum() / y.shape[0])
    xgb = XGBClassifier(objective='binary:logistic', missing=None, seed=42, max_depth=3, colsample_bytree=0.76,
                        gamma=0.03, n_estimators=125, subsample=0.71, learning_rate=0.33,
                        scale_pos_weight=y_train.sum() / y_train.shape[0])
    catboost = CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150, class_weights=class_weights)
    # ACHTUNG: scale_pos_weight=df['Act_class'].sum()/y.shape[0])

    models = {'lr': lr, 'rf': rf, 'gbt': gbt, 'xgb': xgb, 'catboost': catboost,'svm': svm_classifier}
    # lgbm': lgbm,

    # run cv loop

    results_all = pd.DataFrame()

    for key, value in models.items():
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'balanced_accuracy': make_scorer(balanced_accuracy_score),
                   'MCC': make_scorer(matthews_corrcoef),
                   'f1_score': make_scorer(f1_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score)
                   }

        # only 5-fold grouped cross-validation, not 5x5
        cv_fp = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

        results = cross_validate(estimator=value,
                                 X=X_train,
                                 y=y_train,
                                 cv=cv_fp,
                                 scoring=scoring, groups=groups_train)

        result_df = pd.DataFrame.from_dict(results)
        result_df['model'] = key
        results_all = results_all.append(result_df, ignore_index=True)

    # PHNS: adjust tab to make sure that creation of plots is only done once all cross-validation runs are finished
    results_all.to_csv(os.path.join(absolute_path, path, prefix + '_03_model_cv_cluster_TC_'+str(tc)+'.csv'), index=False, sep=';')

    results_all_group = results_all.groupby('model').agg(
        accuracy_mean=pd.NamedAgg(column='test_accuracy', aggfunc=np.mean),
        balanced_accuracy_mean=pd.NamedAgg(column='test_balanced_accuracy', aggfunc=np.mean),
        MCC_mean=pd.NamedAgg(column='test_MCC', aggfunc=np.mean),
        f1_score_mean=pd.NamedAgg(column='test_f1_score', aggfunc=np.mean),
        precision_mean=pd.NamedAgg(column='test_precision', aggfunc=np.mean),
        recall_mean=pd.NamedAgg(column='test_recall', aggfunc=np.mean),
        accuracy_std=pd.NamedAgg(column='test_accuracy', aggfunc=np.std),
        balanced_accuracy_std=pd.NamedAgg(column='test_balanced_accuracy', aggfunc=np.std),
        MCC_std=pd.NamedAgg(column='test_MCC', aggfunc=np.std),
        f1_score_std=pd.NamedAgg(column='test_f1_score', aggfunc=np.std),
        precision_std=pd.NamedAgg(column='test_precision', aggfunc=np.std),
        recall_std=pd.NamedAgg(column='test_recall', aggfunc=np.std),
    ).reset_index()

    results_all_group.to_csv(os.path.join(absolute_path, path, prefix + '_03_model_cv_cluster_group_TC_'+str(tc)+'.csv'), index=False, sep=';')

    y_list = ['test_accuracy', 'test_balanced_accuracy', 'test_MCC', 'test_f1_score', 'test_precision', 'test_recall']

    for i in y_list:
        sns.set_style("whitegrid")
        sns.boxplot(x="model", y=i, data=results_all)
        plt.savefig(os.path.join(absolute_path, path, prefix + '_' + i + '_03_model_cv_cluster_box_TC_'+str(tc)+'.png'))
        # PHNS: avoid overlaying images
        plt.clf()

def model_cv(absolute_path, path, prefix, X_train, y_train):

    #'''stratified k-fold on X_train, cv with lr, rf, gbt, catboost, make cv plots'''
    #return
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=50)
    gbt = GradientBoostingClassifier()
    #lgbm = LGBMClassifier(scale_pos_weight=y.sum() / y.shape[0])
    xgb = XGBClassifier(objective='binary:logistic', missing=None, seed=42, max_depth=3, colsample_bytree=0.76,
                          gamma=0.03, n_estimators=125, subsample=0.71, learning_rate=0.33,
                          scale_pos_weight=y_train.sum() / y_train.shape[0])
    catboost = CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150, class_weights=class_weights)
    # ACHTUNG: scale_pos_weight=df['Act_class'].sum()/y.shape[0])

    models = {'lr': lr, 'rf': rf, 'gbt': gbt, 'xgb': xgb, 'catboost': catboost}
    # lgbm': lgbm,

    # run cv loop

    results_all = pd.DataFrame()

    for key, value in models.items():
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'balanced_accuracy': make_scorer(balanced_accuracy_score),
                   'MCC': make_scorer(matthews_corrcoef),
                   'f1_score': make_scorer(f1_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score)
                   }

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

        results = cross_validate(estimator=value,
                                                 X=X_train,
                                                 y=y_train,
                                                 cv=cv,
                                                 scoring=scoring)

        result_df = pd.DataFrame.from_dict(results)
        result_df['model'] = key
        results_all = results_all.append(result_df, ignore_index=True)

    # PHNS: adjust tab to make sure that creation of plots is only done once all cross-validation runs are finished
    results_all.to_csv(os.path.join(absolute_path, path, prefix + '_03_model_cv.csv'), index=False, sep=';')

    results_all_group = results_all.groupby('model').agg(
        accuracy_mean = pd.NamedAgg(column = 'test_accuracy', aggfunc = np.mean),
        balanced_accuracy_mean= pd.NamedAgg(column = 'test_balanced_accuracy', aggfunc = np.mean),
        MCC_mean= pd.NamedAgg(column = 'test_MCC', aggfunc = np.mean),
        f1_score_mean= pd.NamedAgg(column = 'test_f1_score', aggfunc = np.mean),
        precision_mean= pd.NamedAgg(column = 'test_precision', aggfunc = np.mean),
        recall_mean= pd.NamedAgg(column = 'test_recall', aggfunc = np.mean),
        accuracy_std=pd.NamedAgg(column='test_accuracy', aggfunc=np.std),
        balanced_accuracy_std=pd.NamedAgg(column='test_balanced_accuracy', aggfunc=np.std),
        MCC_std=pd.NamedAgg(column='test_MCC', aggfunc=np.std),
        f1_score_std=pd.NamedAgg(column='test_f1_score', aggfunc=np.std),
        precision_std=pd.NamedAgg(column='test_precision', aggfunc=np.std),
        recall_std=pd.NamedAgg(column='test_recall', aggfunc=np.std),
    ).reset_index()

    results_all_group.to_csv(os.path.join(absolute_path, path, prefix + '_03_model_cv_group.csv'), index=False, sep=';')

    y_list = ['test_accuracy', 'test_balanced_accuracy', 'test_MCC', 'test_f1_score', 'test_precision','test_recall']

    for i in y_list:
        sns.set_style("whitegrid")
        sns.boxplot(x="model", y=i,data=results_all)
        plt.savefig(os.path.join(absolute_path, path, prefix + '_' + i +'_03_model_cv_box.png'))
        # PHNS: avoid overlaying images
        plt.clf()



def train_final_model(absolute_path, path, prefix, model, X_train, X_test, y_train, y_test, X_id):
    print('X_id.head()', X_id.head())
    print('X_train.head()', X_train.head())
    print('X_train.shape', X_train.shape)
    print('len y_train', len(y_train))

    if model == 'catboost':
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        model = CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150, class_weights=class_weights)

        model.fit(X_train, y_train)

        id_list = ['ID', 'SMILES', 'Act_class', 'Train']
        X = X_id.drop(columns = id_list)
        print('X_id.shape:', X_id.shape)
        print('X.shape:', X.shape)

        y_hat = model.predict(X)
        y_hat_test = model.predict(X_test)
        y_proba = model.predict_proba(X)

        X_id['y_hat'] = y_hat
        X_id['proba'] = y_proba[:, 1]
        X_id['proba_inactive'] = y_proba[:, 0]

        print('Shape of y_hat:', y_hat.shape)
        print('Shape of y_proba:', y_proba.shape)

        debug_df = X_id[(X_id['y_hat'] == 1) & (X_id['proba'] < 0.5)]
        print(debug_df[['ID', 'SMILES', 'Act_class', 'Train', 'y_hat', 'proba', 'proba_inactive']])
        print('X_id.shape', X_id.shape)

        model.save_model(os.path.join(absolute_path, path, prefix+'catboost.cbm'),
                                  format="cbm",
                                  export_parameters=None,
                                  pool=None)
        X_id.to_csv(os.path.join(absolute_path, path, prefix+'_03_catboost_predict.csv'), index=False, sep=';')

    elif model == 'gbt':
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_hat = model.predict(X)
        y_hat_test = model.predict(X_test)
        y_proba = model.predict_proba(X)
        y_hat = pd.DataFrame(y_hat)
        y_proba = pd.DataFrame(y_proba)

        X_id['y_hat'] = y_hat
        X_id['proba'] = y_proba[1]
        X_id['Act_class'] = df['Act_class']

        dump(model, prefix + 'gbt.joblib')
        X_id.to_csv(os.path.join(absolute_path, path, prefix + '_03_gbt_predict.csv'), index=False, sep=';')

    #elif model = rf:
    else:
        print('please specify model = catboost, gbt or rf')

    plt.rcParams["figure.figsize"] = (3, 3)
    ax = sns.boxplot(x="y_hat", y="proba", data=X_id)
    plt.savefig(os.path.join(absolute_path, path, prefix +'_03_box_y_hat_vs_proba.png'))

    plt.rcParams["figure.figsize"] = (3, 3)
    ax = sns.boxplot(x="Act_class", y="proba", data=X_id)
    plt.savefig(os.path.join(absolute_path, path, prefix +'_03_box_Act_class_vs_proba.png'))

    plt.rcParams["figure.figsize"] = (3, 3)
    #cm = plot_confusion_matrix(model, X_test, y_test, values_format='d')
    cm = confusion_matrix(y_test, y_hat_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = model.classes_)
    disp.plot()
    plt.savefig(os.path.join(absolute_path, path, prefix +'_03_confusion_matrix.png'))

    # Visualization of ROC-curve
    #metrics.plot_roc_curve(model, X_test, y_test)
    #plt.savefig(os.path.join(absolute_path, path, prefix +'roc_curve.png'))

    # Metrics
    balanced_accuracy = balanced_accuracy_score(y_test, y_hat_test)
    print('balanced_accuracy =' + str(balanced_accuracy))
    f1 = f1_score(y_test, y_hat_test)
    print('f1 =' + str(f1))
    Matthews_scorer = matthews_corrcoef(y_test, y_hat_test)
    print('Matthewes_scorer = ' + str(Matthews_scorer))
    precision = precision_score(y_test, y_hat_test)
    print('precision =' + str(precision))
    recall = recall_score(y_test, y_hat_test)
    print('recall =' + str(recall))

def train_final_model_no_split(absolute_path, path, prefix, model, X_id):
    print('X_id.head()', X_id.head())
    print(list(X_id.columns))
    X_train_no_split = X_id.drop(columns = ['Act_class', 'ID', 'SMILES', 'Activity_percent'], errors='ignore')
    y_train_no_split = X_id['Act_class']

    if model == 'catboost':
        classes = np.unique(y_train_no_split)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_no_split)
        class_weights = dict(zip(classes, weights))
        model = CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150,
                                       class_weights=class_weights)

        model.fit(X_train_no_split, y_train_no_split)

        id_list = ['ID', 'SMILES', 'Act_class', 'Train']
        X = X_id.drop(columns=id_list)
        print('X_id.shape:', X_id.shape)
        print('X.shape:', X.shape)

        y_hat = model.predict(X)
        #y_hat_test = model.predict(X_train_no_split)
        y_proba = model.predict_proba(X)

        X_id['y_hat'] = y_hat
        X_id['proba'] = y_proba[:, 1]
        X_id['proba_inactive'] = y_proba[:, 0]

        print('Shape of y_hat:', y_hat.shape)
        print('Shape of y_proba:', y_proba.shape)

        model.save_model(os.path.join(absolute_path, path, prefix + 'catboost_no_split.cbm'), format="cbm", export_parameters=None, pool=None)
        X_id.to_csv(os.path.join(absolute_path, path, prefix + '_03_catboost_predict_no_split.csv'), index=False, sep=';')

        # Metrics
        balanced_accuracy = balanced_accuracy_score(y_train_no_split, y_hat)
        print('balanced_accuracy =' + str(balanced_accuracy))
        f1 = f1_score(y_train_no_split, y_hat)
        print('f1 =' + str(f1))
        Matthews_scorer = matthews_corrcoef(y_train_no_split, y_hat)
        print('Matthewes_scorer = ' + str(Matthews_scorer))
        precision = precision_score(y_train_no_split, y_hat)
        print('precision =' + str(precision))
        recall = recall_score(y_train_no_split, y_hat)
        print('recall =' + str(recall))



def test_vs_train_scores(absolute_path, path, prefix, model, df, y_hat, y_test, X_test):
    df = df.dropna()
    print(df.shape)
    #plt.rcParams["figure.figsize"] = (3, 3)
    #ax = sns.boxplot(x="y_hat", y="proba", data=df)
    #plt.savefig(os.path.join(absolute_path, path, prefix + '_04_box_y_hat_vs_proba.png'))

    ##plt.rcParams["figure.figsize"] = (3, 3)
    #ax = sns.boxplot(x="Act_class", y="proba", data=df)
    #plt.savefig(os.path.join(absolute_path, path, prefix + '_04_box_Act_class_vs_proba.png'))

    #plt.rcParams["figure.figsize"] = (3, 3)
    #cm = plot_confusion_matrix(model, X_test, y_test, values_format='d')
   # cm = confusion_matrix(y_test, y_hat, labels=model.classes_)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    #plt.savefig(os.path.join(absolute_path, path, prefix + '_04_confusion_matrix.png'))

    # Metrics
    balanced_accuracy = balanced_accuracy_score(y_test, y_hat)
    print('balanced_accuracy =' + str(balanced_accuracy))
    f1 = f1_score(y_test, y_hat)
    print('f1 =' + str(f1))
    Matthews_scorer = matthews_corrcoef(y_test, y_hat)
    print('Matthewes_scorer = ' + str(Matthews_scorer))
    precision = precision_score(y_test, y_hat)
    print('precision =' + str(precision))
    recall = recall_score(y_test, y_hat)
    print('recall =' + str(recall))


def make_cluster_assignment(df, smiles_column, tc):
    ms = df[smiles_column].to_list()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ms]

    # Make sure 'molecules' does not contain any 'None' values
    molecules = [mol for mol in molecules if mol is not None]

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in molecules]

    # Cluster assignments part
    lp = rdSimDivPickers.LeaderPicker()
    thresh = tc  # <- minimum distance between cluster centroids
    picks = lp.LazyBitVectorPick(fps, len(fps), thresh)

    clusters = assignPointsToClusters(picks, fps)

    # 1. Create a list to store cluster assignments for each SMILES
    cluster_assignments = [0] * len(fps)

    for cluster_idx, cluster_members in clusters.items():
        for member_idx in cluster_members:
            cluster_assignments[member_idx] = cluster_idx

    # 3. Create a new DataFrame with SMILES and cluster assignment columns
    df_with_clusters = pd.DataFrame({'SMILES': ms, 'Cluster': cluster_assignments})

    return df_with_clusters

def assignPointsToClusters(picks, fps):
    clusters = defaultdict(list)
    for i, idx in enumerate(picks):
        clusters[i].append(idx)
    sims = np.zeros((len(picks), len(fps)))
    for i in range(len(picks)):
        pick = picks[i]
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
        sims[i, i] = 0
    best = np.argmax(sims, axis=0)
    for i, idx in enumerate(best):
        if i not in picks:
            clusters[idx].append(i)
    return clusters