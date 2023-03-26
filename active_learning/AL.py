import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer,AutoModel
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist,squareform
import heapq 
import pickle

def load_pickle(variable_path):
    '''
    加载通过pickle持久化的变量
    variable_path:变量保存路径
    '''
    f=open(variable_path,'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def dump_pickle(variable_path,variable):
    '''
    将变量持久化
    variable_path:变量保存路径
    variable:待保存变量
    '''
    f=open(variable_path,'wb')
    pickle.dump(variable,f)
    f.close()

# def get_seed(text_featlist):
#     text_featlist=text_featlist/np.linalg.norm(text_featlist, axis=1, keepdims=True)
#     sim = pdist(text_featlist, 'euclidean')
#     sim = squareform(sim)
#     sim_sum=np.sum(sim,axis=1)
#     idx=np.argmin(sim_sum)
    
#     return idx

def get_seed(text_feature,cluster_centers,probs,K = 5):
    distances = - np.array([np.linalg.norm(feature-cluster_centers) for feature in text_feature])
    min_idx =  heapq.nlargest(K, range(len(distances)), distances.take)
    idx=np.argmax(probs[min_idx])
    idx = min_idx[idx]

    return idx


def get_feature(bert_model, device,data_path,collate_function,tokenize_function,batch_size = 128):
    #step1: load pre-train BERT
    bert_model.to(device)

    #step2: Feature extraction using Bert
    raw_datasets = load_dataset("csv", data_files={'data':data_path})
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    dataset_dataloader = DataLoader(tokenized_datasets["data"], shuffle=False, batch_size=batch_size, collate_fn = collate_function)

    features = []
    labels = []
    texts = []
    probs = []
    bert_model.eval()

    for batch in tqdm(dataset_dataloader):
        text = batch.pop("text")
        label = batch.pop("labels")
        label = label.detach().cpu().numpy()
    #     print(labels)

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = bert_model(**batch,output_hidden_states=True)
            logits = outputs.logits
            logits_softmax = torch.softmax(logits, dim=-1)
            logits_softmax = logits_softmax.detach().cpu().numpy()
            for i in range(logits_softmax.shape[0]):
                probs.append(logits_softmax[i,label[i]])

            hidden_states = outputs.hidden_states 
            last_hidden_states = hidden_states[-1]
            sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()

        if features == []:
            features = sample_representation
            labels = label
            texts = text
        else:
            features = np.concatenate([features,sample_representation])
            labels = np.concatenate([labels,label])
            texts =  np.concatenate([texts,text])
    
    dump_pickle('./data/AG_news/base_features.pkl',features)

    return features,texts,labels



def AL_detect(bert_model, device,data_path,class_labels,K_shot,random_state,collate_function,tokenize_function,batch_size = 128 ):
    '''
    active learning
    '''
    #step1: load pre-train BERT
    bert_model.to(device)
    
    #step2: Feature extraction using Bert
    raw_datasets = load_dataset("csv", data_files={'data':data_path})
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    dataset_dataloader = DataLoader(tokenized_datasets["data"], shuffle=False, batch_size=batch_size, collate_fn = collate_function)

    features = []
    labels = []
    texts = []
    probs = []
    bert_model.eval()
    for batch in tqdm(dataset_dataloader):
        text = batch.pop("text")
        label = batch.pop("labels")
        label = label.detach().cpu().numpy()
    #     print(labels)

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = bert_model(**batch,output_hidden_states=True)
            logits = outputs.logits
            logits_softmax = torch.softmax(logits, dim=-1)
            logits_softmax = logits_softmax.detach().cpu().numpy()
            for i in range(logits_softmax.shape[0]):
                probs.append(logits_softmax[i,label[i]])

            hidden_states = outputs.hidden_states 
            last_hidden_states = hidden_states[-1]
            sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()

        if features == []:
            features = sample_representation
            labels = label
            texts = text
        else:
            features = np.concatenate([features,sample_representation])
            labels = np.concatenate([labels,label])
            texts =  np.concatenate([texts,text])
    
    probs = np.array(probs)
    assert len(probs) == len(labels), print('ERROR!')

    #step3: Apply active learning to select representative samples of the  dataset
    select_texts = []
    select_labels = []
    for category in class_labels:
        features_c = features[labels == category]
        labels_c = labels[labels == category]
        texts_c = texts[labels == category]

        clf = KMeans(n_clusters=K_shot, random_state=random_state, max_iter= 1500)
        # y_pred = KMeans(n_clusters=K_shot, random_state=random_state).fit_predict(features_c)
        y_pred = clf.fit_predict(features_c)
        
        cluster_centers = clf.cluster_centers_

        for i in range(K_shot):
            text_feature = features_c[y_pred==i]
            # print(text_feature.shape)
            idx=get_seed(text_feature,cluster_centers,probs)
            # idx=get_seed(text_feature)
    #         LA_idx.append(idx)
            select_texts.append(texts_c[idx])
            select_labels.append(labels_c[idx])
            
    AL_df = pd.DataFrame({'labels':select_labels,'text':select_texts})
    
    return AL_df