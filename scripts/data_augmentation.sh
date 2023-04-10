
# Symtoms 
python3.10 mask_bert.py --cuda 4 --dataset symptoms --task pretrain  --max_length 25
python3.10 mask_bert.py --cuda 4 --dataset symptoms --task AG  --K_shot 2  --novel_batch_size 8 --few_shot_tunning_epochs 150 --max_length 25

#PubMed20K
python3.10 mask_bert.py --cuda 0 --dataset PubMed20k --task pretrain --max_length 40
python3.10 mask_bert.py --cuda 0 --dataset PubMed20k --task AG --max_length 40