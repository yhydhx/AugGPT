from pathlib import Path
import numpy as np
import pandas as pd
import random
import os

from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification,AutoConfig,DataCollatorWithPadding,BertForSequenceClassification
from transformers import get_scheduler
from transformers import  BertForNextSentencePrediction
from transformers import pipeline
from transformers_interpret import SequenceClassificationExplainer
from datasets import load_metric

from build_model.build_ml import seed_everything,tokenize_function,collate_function,build_optimizer,train_one_epoch_without_mask,test_without_mask,get_data_loader
from data_preprocessing.processing import build_few_shot_samples, data_prep


def load_raw_datasets(novel_path,novel_labels,K_shot,novel_few_shot_path,random_seed,is_AL, base_path, novel_test_path):
    few_shot_sample = build_few_shot_samples(novel_path,novel_labels,K_shot,novel_few_shot_path,random_seed,is_AL = False)
    return load_dataset("csv", data_files={'base':base_path,'few_shot_novel':few_shot_sample,'novel_test':novel_test_path})
    

def Pretrain(args):

    args.K_shot = 5
    lr = 2e-5
    #step1: Initialize the model by bert-base-cased
    # # define config
    random_seed = 2022
    seed_everything(random_seed)
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    bert_model = BertForSequenceClassification.from_pretrained(
                    args.model_checkpoint, config=config)
    bert_model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = args.max_length)


    #step2: load dataset and data processing  (without AL)

    data_prep(args.train_path,args.base_path,args.base_labels)
    data_prep(args.train_path,args.novel_path,args.novel_labels)
    data_prep(args.test_path,args.base_test_path,args.base_labels)
    data_prep(args.test_path,args.novel_test_path,args.novel_labels)


    raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,args.K_shot,args.novel_few_shot_path,random_seed,args.is_AL, args.base_path, args.novel_test_path)

    base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,args.base_batch_size,args.novel_batch_size,collate_function,random_seed)

    # # # step2: fine-tunning on base dataset
    seed_everything(random_seed)
    optimizer,lr_scheduler = build_optimizer(bert_model,lr,args.base_tunning_epochs,base_dataloader,True)
    for epoch_init in range(args.base_tunning_epochs):
        bert_model = train_one_epoch_without_mask(bert_model,base_dataloader,optimizer,args.device,lr_scheduler)

    # save model
    bert_model.save_pretrained(args.base_model_save_dir)
    tokenizer.save_pretrained(args.base_model_save_dir)


def data_augmentation(args):
    from build_model.build_ml import seed_everything,tokenize_function,collate_function,build_optimizer,test_without_mask,add_selected_base_dataloader,get_sentences_word_attributions,get_data_loader,get_novel_sample,get_base_neighborhood,get_hybrid_loader,hybrid_train_one_epoch,train_one_epoch_without_mask

    K_shot = 2#args.K_shot
    max_length = args.max_length
    base_batch_size = args.base_batch_size
    novel_batch_size = args.novel_batch_size
    few_shot_tunning_epochs = args.few_shot_tunning_epochs
    with_pos = False
    is_AL = False
    is_continuous = True
    ag_method = args.ag 
    novel_model_save_dir = "./save_models/{}_novel_model{}".format(args.dataset, K_shot)
    novel_few_shot_path = './data/{}/{}shot_search/novel_few_shot_data.csv'.format(args.dataset,args.K_shot)
    result_path = f"./results/{args.dataset}/ag/{args.ag }_constractive_abalation_{K_shot}shot.csv"
    if not os.path.exists(f'./results/{args.dataset}/ag/{args.ag}/'):
        os.makedirs(f'./results/{args.dataset}/ag/{args.ag}/')

    device = args.device
    top_N_ratio = 1
    random_states = args.random_states
    
    params_list = [ 
        # {'is_constractive':False,'with_mask':False,'with_neighborhood':False,'random_select':True,'random_mask':False,'lr':4e-5,'betas' : (0.9,0.999),'with_scheduler' : False,'with_bc':True}, # bert with_bc
        {'is_constractive':True,'with_mask':False,'with_neighborhood':False,'random_select':True,'random_mask':False,'lr':4e-5,'betas' : (0.9,0.999),'with_scheduler' : False,'with_bc':True},  # constractive
    ]


    #step1: Initialize the model by bert-base-cased
    # # define config
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = max_length)


    result_df = pd.DataFrame(columns=['is_constractive','with_mask','with_neighborhood','random_select','random_mask','lr','with_bc','result'])

    for i, params in enumerate(params_list):
        print(params)
        is_constractive = params['is_constractive']
        with_mask = params['with_mask']
        with_neighborhood = params['with_neighborhood']
        random_select = params['random_select']
        random_mask = params['random_mask']
        lr = params['lr']
        betas = params['betas']
        with_scheduler = params['with_scheduler']
        with_bc = params['with_bc']
        
        result = []

        for k, random_state in enumerate(random_states):

            seed_everything(random_state * 10)
            if args.dataset == "PubMed20K":
                K_shot_novel = f"data/ag/{args.ag}/PubMed20K/PubMed20k_novel_few_shot_data{K_shot}_search_{K_shot}_{random_state}.csv"
            else:
                K_shot_novel = f"data/ag/{args.ag}/{args.dataset}/novel_few_shot_data_search_{K_shot}_{random_state}.csv"
            novel_data_dict = get_novel_sample(K_shot_novel)
            raw_datasets = load_dataset("csv", data_files={'base':args.base_path,'few_shot_novel':K_shot_novel,'novel_test':args.novel_test_path})
            base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,base_batch_size,novel_batch_size,collate_function,random_state)
        
            max_acc = 0
            bert_model = BertForSequenceClassification.from_pretrained(
                            args.base_model_save_dir, config=config)
            bert_model.to(device)

            old_text = []

            if with_neighborhood == True:
                neighborhood_data_dict = get_base_neighborhood(bert_model, args.base_path, args.base_labels,K_shot_novel,args.novel_labels, K_shot,collate_function,random_state,random_select = random_select,old_text = old_text)
                hybrid_train_dataloader = get_hybrid_loader(neighborhood_data_dict,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)

            else:
                hybrid_train_dataloader = get_hybrid_loader(None,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)
                    
            optimizer,lr_scheduler = build_optimizer(bert_model,lr,n_epoch = few_shot_tunning_epochs,data_loader = hybrid_train_dataloader,betas = betas,with_scheduler = with_scheduler,with_bc = with_bc)        
            

            #step4 fine-tunning on few-shot novel dataset  (with mask neighborhood samples from the base dataset)
            for few_shot_epoch in range(few_shot_tunning_epochs):
                bert_model.train()

                bert_model = hybrid_train_one_epoch(bert_model,hybrid_train_dataloader,optimizer,top_N_ratio,device,lr_scheduler = lr_scheduler,with_pos=with_pos,is_continuous = is_continuous,is_constractive = is_constractive,with_mask = with_mask,random_mask = random_mask)
                acc = test_without_mask(bert_model,novel_eval_dataloader,device,args.novel_labels)

                if max_acc < acc:
                    print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
                    max_acc = acc
                    bert_model.save_pretrained(novel_model_save_dir)
                    tokenizer.save_pretrained(novel_model_save_dir)
                    if acc == 1:
                        break
                else:
                    print(acc)
                
            bert_model = BertForSequenceClassification.from_pretrained(
                            novel_model_save_dir, config=config)
            bert_model.to(device)
            print('test acc: ',end='')
            acc = test_without_mask(bert_model,novel_test_dataloader,device,args.novel_labels)
            print(acc)
            result.append(acc)

        result_df.loc[i] = [is_constractive,with_mask,with_neighborhood,random_select,random_mask,lr,with_bc,result]
        
        result_df.to_csv(result_path, index = False)   
            
    result_df.to_csv(result_path, index = False) 