import torch

# training params
model_checkpoint = "./save_models/bert-base-uncased"
# model_checkpoint = "./save_models/bert-base-german-cased"

device = torch.device("cuda:0")
K_shot = 2
is_AL = True
is_continuous = False
freeze_mask = False
is_constractive = True
random_select = True
random_mask = False
betas = (0,0)
with_scheduler = False
with_bc = True

random_seeds = [2013,2014,2015]
top_N_ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
mask_layers = [12,10,8,6,4,2,0]
max_length = 80
with_pos = False
top_N = 6  #choose the top_N relevant tokens
base_batch_size = 64
novel_batch_size = 10
base_tunning_epochs = 8
few_shot_tunning_epochs = 150
lr = 2e-5


#  PubMed20k news params
PubMed20k_base_model_save_dir = "./save_models/PubMed20k_base_model"
PubMed20k_novel_model_save_dir = "./save_models/PubMed20k_novel_model"
PubMed20k_train_path = './data/PubMed20k/train.csv'
PubMed20k_test_path = './data/PubMed20k/test.csv'

PubMed20k_base_path = './data/PubMed20k/base_data.csv'
PubMed20k_base_test_path = './data/PubMed20k/base_test_data.csv'
PubMed20k_novel_path = './data/PubMed20k/novel_data.csv'
PubMed20k_novel_few_shot_path = './data/PubMed20k/novel_few_shot_data.csv'
PubMed20k_novel_test_path = './data/PubMed20k/novel_test_data.csv'

PubMed20k_base_labels = [0,1,2]
PubMed20k_novel_labels = [3,4]

PubMed20k_label2id = {'BACKGROUND':0,'OBJECTIVE':1,'METHODS':2,'RESULTS':3,'CONCLUSIONS':4}
PubMed20k_id2label = {0:'BACKGROUND',1:'OBJECTIVE',2:'METHODS',3:'RESULTS',4:'CONCLUSIONS'}






# amazon params
amazon_base_model_save_dir = "./save_models/amazon_base_model"
amazon_novel_model_save_dir = "./save_models/amazon_novel_model"
amazon_train_path = './data/amazon/amazon_train.csv'
amazon_test_path = './data/amazon/amazon_test.csv'

amazon_base_path = './data/amazon/amazon_base_data.csv'
amazon_base_test_path = './data/amazon/amazon_base_test_data.csv'
amazon_novel_path = './data/amazon/amazon_novel_data.csv'
amazon_novel_few_shot_path = './data/amazon/amazon_novel_few_shot_data.csv'
amazon_novel_test_path = './data/amazon/amazon_novel_test_data.csv'

amazon_base_labels = list(range(21))
amazon_novel_labels = list(range(21,24))


amazon_label2id = {'label_' + str(i):i for i in range(24)}
amazon_id2label = {v:k for k,v in amazon_label2id.items()}
