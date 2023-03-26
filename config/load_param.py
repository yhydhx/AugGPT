
def load_parameter(args):

	#common parameter 

	args.model_checkpoint = "./save_models/bert-base-cased"

	args.is_AL = False
	args.is_continuous = False
	args.freeze_mask = False
	args.is_constractive = True
	args.random_select = True
	args.random_mask = False
	args.betas = (0,0)
	args.with_scheduler = False
	args.with_bc = True
	args.random_seeds = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
	args.top_N_ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
	args.mask_layers = [12,10,8,6,4,2,0]
	args.with_pos = False
	args.top_N = 6  #choose the top_N relevant tokens
	args.base_batch_size = 64
	args.novel_batch_size = 10
	args.base_tunning_epochs = 8
	args.few_shot_tunning_epochs = 150
	args.lr = 2e-5

	args.base_model_save_dir = "./save_models/{}_base_model".format(args.dataset)
	args.novel_model_save_dir = "./save_models/{}_novel_model".format(args.dataset)
	args.train_path = './data/{}/train.csv'.format(args.dataset)
	args.test_path = './data/{}/test.csv'.format(args.dataset)

	args.base_path = './data/{}/base_data.csv'.format(args.dataset)
	args.base_test_path = './data/{}/base_test_data.csv'.format(args.dataset)
	args.novel_path = './data/{}/novel_data.csv'.format(args.dataset)
	args.novel_few_shot_path = './data/{}/novel_few_shot_data.csv'.format(args.dataset)
	args.novel_test_path = './data/{}/novel_test_data.csv'.format(args.dataset)


	if args.dataset == "PubMed20k":
		args.base_labels = [0,1,2]
		args.novel_labels = [3,4]

		args.label2id = {'BACKGROUND':0,'OBJECTIVE':1,'METHODS':2,'RESULTS':3,'CONCLUSIONS':4}
		args.id2label = {0:'BACKGROUND',1:'OBJECTIVE',2:'METHODS',3:'RESULTS',4:'CONCLUSIONS'}
	
	elif args.dataset == "PubMed200k":
		args.base_labels = [0,1,2]
		args.novel_labels = [3,4]

		args.label2id = {'BACKGROUND':0,'OBJECTIVE':1,'METHODS':2,'RESULTS':3,'CONCLUSIONS':4}
		args.id2label = {0:'BACKGROUND',1:'OBJECTIVE',2:'METHODS',3:'RESULTS',4:'CONCLUSIONS'}

	elif args.dataset == "symptoms":
		args.base_labels = [0,1,2,3]
		args.novel_labels = [4,5,6]

		args.label2id = {
			'Heart hurts':0,
			'Infected wound':1,
			'Shoulder pain':2,
			'Knee pain':3,
			'Joint pain':4,
			'Acne':5,
			'Cough':6, 
		}

		args.id2label = {v:k  for (k,v) in args.label2id.items()}


	elif args.dataset == "nicta":
		args.dev_path = './data/{}/dev.csv'.format(args.dataset)
		args.novel_val_path = './data/{}/novel_val.csv'.format(args.dataset)
		args.base_labels = [0,1,2]
		args.novel_labels = [3,4]

		args.label2id = {
			'background':0,
			'other':1 ,
			'outcome':2,
			'intervention':3,
			'population':4
		}
		args.id2label = {v:k  for (k,v) in args.label2id.items()}
	
	elif args.dataset == "snippets":
		args.base_labels = [0,1,2,3]
		args.novel_labels = [4,5,6,7]

		args.label2id = {
			'business':0,
			'computers':1,
			'culture-arts-entertainment':2,
			'education-science':3,
			'engineering':4,
			'health':5,
			'politics-society':6,
			'sports':7
		}

		args.id2label = {v:k  for (k,v) in args.label2id.items()}

	elif args.dataset == "dbpedia14":
		args.base_labels = [0,1,2,3,4,5,6,7]
		args.novel_labels = [8,9,10,11,12,13]

		args.label2id = {'Company':0,'EducationalInstitution':1,'Artist':2,'Athlete':3,'OfficeHolder':4,'MeanOfTransportation':5,'Building':6,'NaturalPlace':7,'Village':8,'Animal':9,'Plant':10,'Album':11,'Film':12,'WrittenWork':13}
		args.id2label = {v:k for k,v in args.label2id.items()}

	elif args.dataset == "AG_news":

		args.base_labels = [0,1]
		args.novel_labels = [2,3]

		args.label2id = {'World':0,'Sports':1,'Business':2,'Sci/Tech':3}
		args.id2label = {0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}