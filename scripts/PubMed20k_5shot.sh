#  fine-tunning model on base dataset 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 0 --K_shot 5 --base_batch_size 64 --novel_batch_size 8  --max_length 40



#BERT_CNN model 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 1 --K_shot 5  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

#CPFT model 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 2 --K_shot 5 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

#FPT BERT model
python mask_bert.py --cuda 4 --dataset PubMed20k --task 3 --K_shot 5 --base_batch_size 64 --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

#Reinit BERT Model 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 4 --K_shot 5  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

#NSP BERT 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 6 --K_shot 5  --novel_batch_size 8 --few_shot_tunning_epochs 10 --max_length 40

#SNFT BERT 
python mask_bert.py --cuda 4 --dataset PubMed20k --task 7 --K_shot 5  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

#Mask BERT
python mask_bert.py --cuda 4 --dataset PubMed20k --task 5 --K_shot 5  --novel_batch_size 8 --few_shot_tunning_epochs 50 --max_length 40

