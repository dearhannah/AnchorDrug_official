import sys, os, re, argparse, warnings, datetime, random, math, pickle, time
import numpy as np
import pandas as pd
import torch
from utils import get_dataset, get_net, get_strategy

torch.set_printoptions(profile='full')
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import arguments
from parameters import *
from utils import *

# parameters
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
SEED = args_input.seed
os.environ['TORCH_HOME']='./basicmodel'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# # fix random seed
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.backends.cudnn.enabled  = True
# torch.backends.cudnn.benchmark= True

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#recording
timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
sys.stdout = Logger(f'./logfile/{DATA_NAME}_{args_input.cell}_{STRATEGY_NAME}_{str(NUM_QUERY)}_{str(NUM_INIT_LB)}_{str(args_input.quota)}_{timestamp}.txt')
warnings.filterwarnings('ignore')

# start experiment

iteration = args_input.iteration

all_acc = []
all_f1 = []
acq_time = []

# repeate # iteration trials

start = time.time()
while (iteration > 0): 
	print(f'------------------iteration:{iteration}------------------')
	iteration = iteration - 1
	# data, network, strategy
	args_task = args_pool[DATA_NAME]
	args_task['cell'] = [args_input.cell]
	dataset = get_dataset(args_input.dataset_name, args_task)				# load dataset
	net_all = []
	for cell in dataset.cell_list:
		args_task['cell'] = cell
		net = get_net(args_input.dataset_name, args_task, device)			# load network
		net_all.append(net)
	args_task['cell'] = dataset.cell_list
	strategy = get_strategy(args_input.ALstrategy, dataset, net_all, args_input, args_task)  # load strategy

	# generate initial labeled pool
	dataset.initialize_labels(args_input.initseed)
	# record acc performance
	acc = np.zeros((NUM_ROUND+1, 1))
	f1 = np.zeros((NUM_ROUND+1, 1))
	# recored prediction and label
	YandPred = {}
	YandPred['label'] = dataset.Y_val[0]
	# print info
	print(DATA_NAME)
	# print('RANDOM SEED {}'.format(SEED))
	# print(type(strategy).__name__)

	print('Round 0:')
	for i in range(len(dataset.cell_list)):
		# print(dataset.cell_list[i])
		preds = strategy.predict(i, dataset.get_test_data(dataID=i))
		acc[0][i] = dataset.cal_test_acc(preds, i)
		print('testing accuracy {}'.format(acc[0][i]))
		f1[0][i] = dataset.cal_test_f1(preds, i)
		print('testing F1 {}'.format(f1[0][i]))
		print('comfusin matrix')
		cm = dataset.cal_test_confusion(preds, i)
		print(cm)
		print(*cm.reshape(9), sep = ", ")
		YandPred[0] = preds
	
	# round 1 to rd
	for rd in range(1, NUM_ROUND+1):
		print('Round {}:'.format(rd))
		print(f'current used time: {time.time() - start}')
		# query
		q_idxs = strategy.query(NUM_QUERY)
		# update
		strategy.update(q_idxs)
		#train
		strategy.train()
		# round rd accuracy
		for i in range(len(dataset.cell_list)):
			# print(dataset.cell_list[i])
			preds = strategy.predict(i, dataset.get_test_data(dataID=i))
			acc[rd][i] = dataset.cal_test_acc(preds, i)
			print('testing accuracy {}'.format(acc[rd][i]))
			f1[rd][i] = dataset.cal_test_f1(preds, i)
			print('testing F1 {}'.format(f1[rd][i]))
			print('comfusin matrix')
			cm = dataset.cal_test_confusion(preds, i)
			print(cm)
			print(*cm.reshape(9), sep = ", ")
			YandPred[rd] = preds
	idx, smiles = dataset.get_labeled_drugs()
	# [print(s) for s in smiles]
	
	# print results
	# print('SEED {}'.format(SEED))
	print(type(strategy).__name__)
	print('acc:', acc)
	print('f1:', f1)
	all_acc.append(acc)
	all_f1.append(f1)

	#save drug
	drug_path = f'./druglist/{DATA_NAME}_{args_input.cell}_{STRATEGY_NAME}_{str(NUM_QUERY)}_{str(NUM_INIT_LB)}_{str(args_input.quota)}_{timestamp}_{iteration}.pkl'
	with open(drug_path, 'wb') as f:
		pickle.dump(smiles, f)
	# save preds
	preds_path = f'./preds/{DATA_NAME}_{args_input.cell}_{STRATEGY_NAME}_{str(NUM_QUERY)}_{str(NUM_INIT_LB)}_{str(args_input.quota)}_{timestamp}_{iteration}.pkl'
	with open(preds_path, 'wb') as f:
		pickle.dump(YandPred, f)
	preds_csv_path = f'./preds/{DATA_NAME}_{args_input.cell}_{STRATEGY_NAME}_{str(NUM_QUERY)}_{str(NUM_INIT_LB)}_{str(args_input.quota)}_{timestamp}_{iteration}.csv'
	pd.DataFrame.from_dict(YandPred).to_csv(preds_csv_path,index=False)

print(f'!!!!!total used time: {time.time() - start}')
#save F1,acc
res_path = f'./results/{DATA_NAME}_{args_input.cell}_{STRATEGY_NAME}_{str(NUM_QUERY)}_{str(NUM_INIT_LB)}_{str(args_input.quota)}_{timestamp}.pkl'
with open(res_path, 'wb') as f:
	pickle.dump((all_acc,all_f1), f)
