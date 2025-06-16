# CS-SMOL-TPO-model-publication

## **Overview**
This repository contains the codebase for the scientific publication titled "De-risking future agrochemicals before they are made: Large-scale in vitro screening for in silico modeling of Thyroperoxidase inhibition". It includes a modeling pipeline for preprocessing, training, cross-validating, and utilizing a TPO-model, alongside a Jupyter notebook that allows to simulate the effect of applying such model to a realistic screening/deselection usecase. The aim is to facilitate the replication of our results and provide a foundation for further research and development.

## Correction Notice
In the original paper, we implied that class weights were applied to all tested models. However, this was not the case for GBT, RF, and LR. To address this, we have provided an updated training script:

- **`03_Train_model_with_correct_weights.py`**: This script implements class weights for GBT, RF, and LR, resulting in improved metrics.

- **`03_Train_model.py`**: Use this original script to reproduce the results as published in the paper.

 ### Performance Metrics Comparison

The corrected implementation yields the following changes:

- GBT now produces comparable results to catboost in in Matthews Correlation Coefficient (MCC) recall and other metrics.

- The detailed performance metrics for all models are included in the repository (code/cv_comparison_class_weights.xlsx) for reference.

- All other interpretations and conclusion in our paper are unchanged

### **Installation**
- **Clone the repository:**
```
git clone https://github.com/NinaKausch/CS-SMOL-TPO_model_publication
```

- **Install the required dependencies:**
```
pip install -r requirements.txt
```
 
### **Usage**

- **Preprocessing pipeline and feature generation:**
```
python 02_Make_RDKIT_features.py --path  <your path> --input <your_input_file>.csv --Threshold 1 --Train
```

- **Training the Model**

with adjusted class weights for all models run

```
python 03_Train_model_with_correct_weights.py  <your path> --input <your_input_file>_02_train_test.csv --model catboost --cv --cluster --TC 0.2
```
 
to reproduce the published results run 

```
python 03_Train_model.py  <your path> --input <your_input_file>_02_train_test.csv --model catboost --cv --cluster --TC 0.2
```


- **Inference**
, to run an inference use:
```
python 04_Predict.py  <your path> --input <your_input_file>_orig_feat.csv --trained_model <your_model>.cbm --features <your_input_file>_02_train_testcatboost_03_required_features.csv --test_set
```

you can also set parameters for model preprocessing, training and inference in a bash-script as exemplified under examples. 

### **Jupyter Notebook**
, to explore the effect of different modeling approaches:
- **Launch Jupyter Notebook:**
```
Model_hitrate_simulation_precision_recall.ipynb
```
