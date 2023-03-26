# Step1: Pretrain model 
python3.10 mask_bert.py --cuda 0 --dataset PubMed20k --task pretrain --max_length 30

#docker run -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all test1


#AUGBERT  with differnt K-shot 
python3.10 mask_bert.py --cuda 0 --dataset PubMed20k --task AG --K_shot 2 --max_length 40
python3.10 mask_bert.py --cuda 1 --dataset PubMed20k --task AG --K_shot 3 --max_length 40
python3.10 mask_bert.py --cuda 2 --dataset PubMed20k --task AG --K_shot 5 --max_length 40
python3.10 mask_bert.py --cuda 3 --dataset PubMed20k --task AG --K_shot 8 --max_length 40
python3.10 mask_bert.py --cuda 4 --dataset PubMed20k --task AG --K_shot 10 --max_length 40
