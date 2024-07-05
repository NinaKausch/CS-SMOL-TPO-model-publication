source ~/.bashrc
conda activate base
python /home/gfuis/CS-SMOL-TPO_model_publication/code/01_Standardize.py \
       --path data \
       --infile TPO_new_holdout_2021.csv \
       --sep '\t' \
       --id_col 'ID' \
       --smiles_col 'SMILES' \
       --test_set \
       --activity_col 'Activity' \
       --get_parent
 





	
