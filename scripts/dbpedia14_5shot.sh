#  fine-tunning model on base dataset 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 0 --K_shot 5 --base_batch_size 64 --novel_batch_size 32  --max_length 100



#BERT_CNN model 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 1 --K_shot 5  --novel_batch_size 32 --few_shot_tunning_epochs 50 --max_length 100

#CPFT model 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 2 --K_shot 5 --base_batch_size 64 --novel_batch_size 32 --few_shot_tunning_epochs 150 --max_length 100

#FPT BERT model
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 3 --K_shot 5 --base_batch_size 64 --novel_batch_size 32 --few_shot_tunning_epochs 150 --max_length 100

#Reinit BERT Model 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 4 --K_shot 5  --novel_batch_size 32 --few_shot_tunning_epochs 50 --max_length 100

#NSP BERT 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 6 --K_shot 5  --novel_batch_size 32 --few_shot_tunning_epochs 10 --max_length 100

#SNFT BERT 
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 7 --K_shot 5  --novel_batch_size 32 --few_shot_tunning_epochs 50 --max_length 100

#Mask BERT
python mask_bert.py --cuda 4 --dataset dbpedia14 --task 5 --K_shot 5  --novel_batch_size 32 --few_shot_tunning_epochs 50 --max_length 100

