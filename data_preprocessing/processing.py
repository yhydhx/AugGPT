import pandas as  pd

from active_learning.AL import AL_detect
from build_model.build_ml import collate_function,tokenize_function
from config import config_params 
import os

global device,max_length,model_checkpoint
device = config_params.device
max_length = config_params.max_length
model_checkpoint = config_params.model_checkpoint





def data_prep(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path,sep="\t")
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)

#pubmed21k 

def PubMed21k_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    output_file = open(novel_data_path, 'w')
    count = -1
    for line in open(train_path):
        count += 1
        if count == 0:
            output_file.write(line)
            continue
        text, labels = line.strip().split("\t")
        if int(labels) in novel_labels:
            output_file.write(line)
    return 

def PubMed21k_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path,sep="\t")
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)


#AG news
def AG_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path)
    data_df['text'] = data_df['Title'] + ' . ' +  data_df['Description']
    data_df['labels'] = data_df['Class Index']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)




def AG_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    data_df = pd.read_csv(train_path)
    data_df['text'] = data_df['Title'] + ' . ' +  data_df['Description']
    data_df['labels'] = data_df['Class Index']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in novel_labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(novel_data_path,index = False)

#dbpedia14
def dbpedia14_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path)
    data_df['text'] = data_df['Title'] + ' . ' +  data_df['Description']
    data_df['labels'] = data_df['Class Index']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)
    
    
def dbpedia14_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    data_df = pd.read_csv(train_path)
    data_df['text'] = data_df['Title'] + ' . ' +  data_df['Description']
    data_df['labels'] = data_df['Class Index']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in novel_labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(novel_data_path,index = False)
    

#gnad10
def gnad10_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path)
    data_df['labels'] = data_df['label']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)
    
    
def gnad10_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    data_df = pd.read_csv(train_path)
    data_df['labels'] = data_df['label']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in novel_labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(novel_data_path,index = False)
    

#amazon
def amazon_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path)
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)
    
    
def amazon_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    data_df = pd.read_csv(train_path)
    data_df = data_df[['labels','text']]
    select_idx = [True if label in novel_labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(novel_data_path,index = False)


#pitt
def pitt_data_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path)
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)
    
    
def pitt_novel_data_processing(train_path,novel_data_path,novel_labels):
    #select the 
    data_df = pd.read_csv(train_path)
    data_df = data_df[['labels','text']]
    select_idx = [True if label in novel_labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(novel_data_path,index = False)


def build_few_shot_samples(novel_data_path,novel_labels,K_shot,novel_few_shot_path,random_state,is_AL = True):

    if is_AL:
        # few_shot_df = AL_detect(model_checkpoint, device,novel_data_path,novel_labels,K_shot,random_state,collate_function,tokenize_function )
        novel_few_shot_path = novel_few_shot_path[:-4] + '_' + str(K_shot) + '_' + str(random_state) + novel_few_shot_path[-4:]
        few_shot_df =  pd.read_csv(novel_few_shot_path,sep="\t")
        # We have obtained activate learning samples with coreset

    else:    

        data_df = pd.read_csv(novel_data_path)
        few_shot_df = None
        for label in novel_labels:
            tmp = pd.DataFrame(data_df[data_df.labels == label])
            tmp = tmp.sample(n=K_shot,random_state = random_state)
            if type(few_shot_df) == type(None):
                few_shot_df = tmp
            else:
                few_shot_df = pd.concat([few_shot_df,tmp], ignore_index=True)
        few_shot_df = few_shot_df.reset_index(drop=True)
        
        novel_few_shot_path = novel_few_shot_path[:-4] + '_' + str(K_shot) + '_' + str(random_state) + novel_few_shot_path[-4:]
        if not os.path.exists(novel_few_shot_path):
            few_shot_df.to_csv(novel_few_shot_path,index = False)
        return novel_few_shot_path

