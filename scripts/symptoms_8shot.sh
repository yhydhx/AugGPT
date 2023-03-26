#  fine-tunning model on base dataset 
python mask_bert.py --cuda 4 --dataset symptoms --task 0 --K_shot 8 --base_batch_size 32 --novel_batch_size 8  --max_length 25



#BERT_CNN model 
python mask_bert.py --cuda 4 --dataset symptoms --task 1 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

#CPFT model 
python mask_bert.py --cuda 4 --dataset symptoms --task 2 --K_shot 8 --base_batch_size 32 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

#FPT BERT model
python mask_bert.py --cuda 4 --dataset symptoms --task 3 --K_shot 8 --base_batch_size 32 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

#Reinit BERT Model 
python mask_bert.py --cuda 4 --dataset symptoms --task 4 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

#NSP BERT 
python mask_bert.py --cuda 4 --dataset symptoms --task 6 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 10 --max_length 25

#SNFT BERT 
python mask_bert.py --cuda 4 --dataset symptoms --task 7 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 25

#Mask BERT
python mask_bert.py --cuda 4 --dataset symptoms --task 5 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

