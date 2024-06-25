
source ~/.bashrc

conda activate base

python /home/gfuis/CS-SMOL-TPO_model_publication/code/01_Standardize.py \
       --path  data \
       --infile S2_Result_list.txt \
       --sep ','\
       --id_col 'none'\
       --smiles_col 'Structure (as SMILES)'\
       --test_set True\
       --activity_col 'activity class'\  
 





	
