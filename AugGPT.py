import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
from config.load_param import load_parameter
from methods import *

parser = argparse.ArgumentParser(
    description='Pytorch Mask Bert Training')

parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dataset', default='PubMed20k', type=str)
parser.add_argument('--task', default="BERT", type=str)
parser.add_argument('--K_shot', default=2, type=int)
parser.add_argument('--base_batch_size', default=64, type=int)
parser.add_argument('--novel_batch_size', default=8, type=int)
parser.add_argument('--few_shot_tunning_epochs', default=150, type=int)
parser.add_argument('--max_length', default=60, type=int)


args = parser.parse_args()

load_parameter(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
cuda = True if torch.cuda.is_available() else False
args.device = torch.device('cuda' if cuda else 'cpu')

if args.task == "pretrain":
    #pretrain the model 
    Pretrain(args)
elif args.task == "AG":
    #for ag_method in ["DeleteCharAugmentation", "InsertCharAugmentation", "InsertWordByGoogleNewsEmbeddings", "KeyboardAugmentation", "OCRAugmentation", "SpellingAugmentation", "SubstituteCharAugmentation","SubstituteWordByGoogleNewsEmbeddings","SwapCharAugmentation"]:
    for ag_method in os.listdir("./data/ag/"):
        print("start data augmentation:", ag_method)
        args.ag = ag_method
        data_augmentation(args)
