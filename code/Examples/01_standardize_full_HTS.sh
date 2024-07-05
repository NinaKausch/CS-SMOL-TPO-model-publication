source ~/.bashrc
conda activate base
python /home/gfuis/CS-SMOL-TPO_model_publication/code/01_Standardize.py \
       --path data \
       --infile TPO_full_HTS.csv \
       --sep '\t' \
       --id_col 'ID' \
       --smiles_col 'raw_SMILES' \
       --test_set \
       --activity_col 'Activity' \
       --get_parent
 





	
