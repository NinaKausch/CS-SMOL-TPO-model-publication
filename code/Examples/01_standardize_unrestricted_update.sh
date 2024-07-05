source ~/.bashrc
conda activate base
python /home/gfuis/CS-SMOL-TPO_model_publication/code/01_Standardize.py \
       --path data \
       --infile S2_Result_list_updated.txt \
       --sep ',' \
       --id_col 'random_id' \
       --smiles_col 'Structure (as SMILES)' \
       --test_set \
       --activity_col 'activity class' \
       --get_parent
 





	
