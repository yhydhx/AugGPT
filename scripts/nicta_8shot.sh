#  fine-tunning model on base dataset 
python mask_bert.py --cuda 4 --dataset nicta --task 0 --K_shot 8 --base_batch_size 64 --novel_batch_size 8  --max_length 55



#BERT_CNN model 
python mask_bert.py --cuda 4 --dataset nicta --task 1 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 55

#CPFT model 
python mask_bert.py --cuda 4 --dataset nicta --task 2 --K_shot 8 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 55

#FPT BERT model
python mask_bert.py --cuda 4 --dataset nicta --task 3 --K_shot 8 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 55

#Reinit BERT Model 
python mask_bert.py --cuda 4 --dataset nicta --task 4 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 55

#NSP BERT 
python mask_bert.py --cuda 4 --dataset nicta --task 6 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 10 --max_length 55

#SNFT BERT 
python mask_bert.py --cuda 4 --dataset nicta --task 7 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 55

#Mask BERT
python mask_bert.py --cuda 4 --dataset nicta --task 5 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 55

