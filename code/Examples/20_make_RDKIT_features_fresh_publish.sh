
source ~/.bashrc

conda activate base

python /home/gfuis/CS-SMOL-TECHFU-TPO_model/code/02_Make_RDKIT_features.py \
       --path  data \
       --input TPO_HTS_PUBLISH_refresh.csv \
       --Threshold 1 \
       --Train \
       #--Bayer_Models \
       #--Bayer_model_file _SMILES_merged.csv \





	
