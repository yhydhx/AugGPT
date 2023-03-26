#  fine-tunning model on base dataset 
python mask_bert.py --cuda 4 --dataset AG_news --task 0 --K_shot 8 --base_batch_size 64 --novel_batch_size 8  --max_length 80



#BERT_CNN model 
python mask_bert.py --cuda 4 --dataset AG_news --task 1 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 80

#CPFT model 
python mask_bert.py --cuda 4 --dataset AG_news --task 2 --K_shot 8 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 80

#FPT BERT model
python mask_bert.py --cuda 4 --dataset AG_news --task 3 --K_shot 8 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 80

#Reinit BERT Model 
python mask_bert.py --cuda 4 --dataset AG_news --task 4 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 80

#NSP BERT 
python mask_bert.py --cuda 4 --dataset AG_news --task 6 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 80

#SNFT BERT 
python mask_bert.py --cuda 4 --dataset AG_news --task 7 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 10 --max_length 80

#Mask BERT
python mask_bert.py --cuda 4 --dataset AG_news --task 5 --K_shot 8  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 80

