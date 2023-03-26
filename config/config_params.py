import torch

# training params
model_checkpoint = "./save_models/bert-base-cased"
# model_checkpoint = "./save_models/bert-base-german-cased"

device = torch.device("cuda:0")
K_shot = 5
is_AL = True
is_continuous = False
freeze_mask = False
is_constractive = True
random_select = True
random_mask = False
betas = (0,0)
with_scheduler = False
with_bc = True

random_seeds = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
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



#  AG news params
AG_base_model_save_dir = "./save_models/AG_base_model"
AG_novel_model_save_dir = "./save_models/AG_novel_model"
AG_train_path = './data/AG_news/train.csv'
AG_test_path = './data/AG_news/test.csv'

AG_base_path = './data/AG_news/AG_base_data.csv'
AG_base_test_path = './data/AG_news/AG_base_test_data.csv'
AG_novel_path = './data/AG_news/AG_novel_data.csv'
AG_novel_few_shot_path = './data/AG_news/AG_novel_few_shot_data.csv'
AG_novel_test_path = './data/AG_news/AG_novel_test_data.csv'

AG_base_labels = [0,1]
AG_novel_labels = [2,3]

AG_label2id = {'World':0,'Sports':1,'Business':2,'Sci/Tech':3}
AG_id2label = {0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}




# dbpedia14 params
dbpedia14_base_model_save_dir = "./save_models/dbpedia14_base_model"
dbpedia14_novel_model_save_dir = "./save_models/dbpedia14_novel_model"
dbpedia14_train_path = './data/dbpedia14/dbpedia14_train.csv'
dbpedia14_test_path = './data/dbpedia14/dbpedia14_test.csv'

dbpedia14_base_path = './data/dbpedia14/dbpedia14_base_data.csv'
dbpedia14_base_test_path = './data/dbpedia14/dbpedia14_base_test_data.csv'
dbpedia14_novel_path = './data/dbpedia14/dbpedia14_novel_data.csv'
dbpedia14_novel_few_shot_path = './data/dbpedia14/dbpedia14_novel_few_shot_data.csv'
dbpedia14_novel_test_path = './data/dbpedia14/dbpedia14_novel_test_data.csv'

dbpedia14_base_labels = [0,1,2,3,4,5,6,7]
dbpedia14_novel_labels = [8,9,10,11,12,13]

dbpedia14_label2id = {'Company':0,'EducationalInstitution':1,'Artist':2,'Athlete':3,'OfficeHolder':4,'MeanOfTransportation':5,'Building':6,'NaturalPlace':7,'Village':8,'Animal':9,'Plant':10,'Album':11,'Film':12,'WrittenWork':13}
dbpedia14_id2label = {v:k for k,v in dbpedia14_label2id.items()}




# gnad10 params
gnad10_model_save_dir = "./save_models/gnad10_base_model"
gnad10_train_path = './data/gnad10/gnad10_train.csv'
gnad10_test_path = './data/gnad10/gnad10_test.csv'

gnad10_base_path = './data/gnad10/gnad10_base_data.csv'
gnad10_base_test_path = './data/gnad10/dgnad10_base_test_data.csv'
gnad10_novel_path = './data/gnad10/gnad10_novel_data.csv'
gnad10_novel_few_shot_path = './data/gnad10/gnad10_novel_few_shot_data.csv'
gnad10_novel_test_path = './data/gnad10/gnad10_novel_test_data.csv'

gnad10_base_labels = [0,1,2,3,4,5]
gnad10_novel_labels = [6,7,8]

gnad10_label2id = {'Web':0,'Panorama':1,'International':2,'Wirtschaft':3,'Sport':4,'Inland':5,'Etat':6,'Wissenschaft':7,'Kultur':8}
gnad10_id2label = {v:k for k,v in gnad10_label2id.items()}




# pitt params
pitt_base_model_save_dir = "./save_models/pitt_base_model"
pitt_novel_model_save_dir = "./save_models/pitt_novel_model"
pitt_train_path = './data/pitt/pitt_train.csv'
pitt_test_path = './data/pitt/pitt_test.csv'

pitt_base_path = './data/pitt/pitt_base_data.csv'
pitt_base_test_path = './data/pitt/pitt_base_test_data.csv'
pitt_novel_path = './data/pitt/pitt_novel_data.csv'
pitt_novel_few_shot_path = './data/pitt/pitt_novel_few_shot_data.csv'
pitt_novel_test_path = './data/pitt/pitt_novel_test_data.csv'

pitt_base_labels = [0,1]
pitt_novel_labels = [2,3]

pitt_label2id = {'Control':0,'ProbableAD':1,'MCI':2,'PossibleAD':3}
pitt_id2label = {v:k for k,v in pitt_label2id.items()}


# food params
food_base_model_save_dir = "./save_models/food_base_model"
food_novel_model_save_dir = "./save_models/food_novel_model"
food_train_path = './data/food/food_train.csv'
food_test_path = './data/food/food_test.csv'

food_base_path = './data/food/food_base_data.csv'
food_base_test_path = './data/food/food_base_test_data.csv'
food_novel_path = './data/food/food_novel_data.csv'
food_novel_few_shot_path = './data/food/food_novel_few_shot_data.csv'
food_novel_test_path = './data/food/food_novel_test_data.csv'

food_base_labels = list(range(20))
food_novel_labels = list(range(20,34))

food_label2id = {'Fruit-flavored beverage, dry concentrate, with sugar, not reconstituted': 0,
 'Tea, NS as to type, presweetened with sugar': 1,
 'Beer': 2,
 'Fruit leather and fruit snacks candy': 3,
 'Soft drink, fruit-flavored, caffeine free': 4,
 'Tortilla, flour (wheat)': 5,
 'Fruit juice drink': 6,
 'White potato chips, regular cut': 7,
 'Roll, white, soft': 8,
 'Pork sausage, fresh, bulk, patty or link, cooked': 9,
 'Salty snacks, corn or cornmeal base, tortilla chips': 10,
 'Gumdrops': 11,
 'Salsa, red, commercially-prepared': 12,
 'Water, bottled, unsweetened': 13,
 'Soup, ramen noodle, any flavor, dry': 14,
 'Gatorade Thirst Quencher sports drink': 15,
 'Pizza with meat, prepared from frozen, thin crust': 16,
 'Bread, white': 17,
 'Egg, whole, raw': 18,
 'White potato chips, ruffled, rippled, or crinkle cut': 19,
 'Coffee, dry instant powder, regular': 20,
 'Chewing gum, sugarless': 21,
 'Turkey or chicken breast, prepackaged or deli, luncheon meat': 22,
 'Ice cream, regular, flavors other than chocolate': 23,
 'Yogurt, fruit variety, lowfat milk': 24,
 'Yogurt, fruit variety, NS as to type  of milk': 25,
 'Spices, poultry seasoning': 26,
 'Pork, cured, bacon, unprepared': 27,
 'Spaghetti sauce, meatless': 28,
 'Ham, sliced, prepackaged or deli, luncheon meat': 29,
 'Carbonated water, unsweetened': 30,
 'Hard candy': 31,
 'Macaroni and cheese dinner with dry sauce mix, boxed, uncooked': 32,
 'Macaroni, dry, enriched': 33}

food_id2label = {v:k for k,v in food_label2id.items()}




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

amazon_base_labels = list(range(15))
amazon_novel_labels = list(range(15,24))


amazon_label2id = {'label_' + str(i):i for i in range(24)}
amazon_id2label = {v:k for k,v in amazon_label2id.items()}
