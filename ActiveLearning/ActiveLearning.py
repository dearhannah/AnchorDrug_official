import sys, os, re, argparse, warnings, datetime, random, math, pickle
import numpy as np
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

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= True

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#recording
sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME + '_' + args_input.cell + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_log.txt')
warnings.filterwarnings('ignore')

# start experiment

iteration = args_input.iteration

all_acc = []
all_f1 = []
acq_time = []

# repeate # iteration trials
while (iteration > 0): 
	iteration = iteration - 1
	# data, network, strategy
	args_task = args_pool[DATA_NAME]
	args_task['cell'] = args_input.cell
	dataset = get_dataset(args_input.dataset_name, args_task)				# load dataset
	net_all = []
	for gene in dataset.genelist:
		args_task['gene'] = gene
		net = get_net(args_input.dataset_name, args_task, device)			# load network
		net_all.append(net)
	strategy = get_strategy(args_input.ALstrategy, dataset, net_all, args_input, args_task)  # load strategy

	start = datetime.datetime.now()
	# generate initial labeled pool
	dataset.initialize_labels(args_input.initseed)
	#record acc performance
	acc = np.zeros((NUM_ROUND, 5))
	f1 = np.zeros((NUM_ROUND, 5))
		
	# print info
	print(DATA_NAME)
	print('RANDOM SEED {}'.format(SEED))
	print(type(strategy).__name__)

	print('Round 0:')
	for i in range(5):
		print(dataset.genelist[i])
		preds = strategy.predict(i, dataset.get_test_data(dataID=i))
		acctmp = dataset.cal_test_acc(preds, i)
		print('testing accuracy {}'.format(acctmp))
		f1tmp = dataset.cal_test_f1(preds, i)
		print('testing F1 {}'.format(f1tmp))
	
	# round 1 to rd
	for rd in range(0, NUM_ROUND):
		print('Round {}'.format(rd))
		# query
		q_idxs = strategy.query(NUM_QUERY)
		# update
		strategy.update(q_idxs)
		#train
		strategy.train()
		# round rd accuracy
		for i in range(5):
			print(dataset.genelist[i])
			preds = strategy.predict(i, dataset.get_test_data(dataID=i))
			acc[rd][i] = dataset.cal_test_acc(preds, i)
			print('testing accuracy {}'.format(acc[rd][i]))
			f1[rd][i] = dataset.cal_test_f1(preds, i)
			print('testing F1 {}'.format(f1[rd][i]))
		idx, smiles = dataset.get_labeled_drugs()
		print(idx)
		[print(s) for s in smiles]
	
	# print results
	print('SEED {}'.format(SEED))
	print(type(strategy).__name__)
	print('acc:', acc)
	print('f1:', f1)
	all_acc.append(acc)
	all_f1.append(f1)
	
	#save model
	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	model_path = './modelpara/'+timestamp + DATA_NAME + '_' + args_input.cell + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '.params'
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds),3))
	torch.save(strategy.get_model().state_dict(), model_path)
	#save drug
	drug_path = './druglist/'+ timestamp + DATA_NAME + '_' + args_input.cell + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '.pkl'
	with open(drug_path, 'wb') as f:
		pickle.dump(smiles, f)
		
#save F1,acc
res_path = './results/'+ DATA_NAME + '_' + args_input.cell + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '.pkl'
with open(res_path, 'wb') as f:
	pickle.dump((all_acc,all_f1), f)
		
	
# # cal mean & standard deviation
# acc_m = []
# f1_m = []
# file_name_res_tot = DATA_NAME + '_' + args_input.cell + '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res_tot.txt'
# file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res_tot),'w')
# file_res_tot.writelines('dataset: {}'.format(DATA_NAME) + '\n')
# file_res_tot.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
# file_res_tot.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
# file_res_tot.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
# file_res_tot.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
# file_res_tot.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
# file_res_tot.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
# file_res_tot.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')

# # result
# for i in range(len(all_acc)):
# 	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
# 	f1_m.append(get_aubc(args_input.quota, NUM_QUERY, all_f1[i]))
# 	print(str(i)+' acc: '+str(acc_m[i])+' f1: '+str(f1_m[i]))
# 	file_res_tot.writelines(str(i)+' acc: '+str(acc_m[i])+' f1: '+str(f1_m[i])+'\n')
# mean_acc, stddev_acc = get_mean_stddev(acc_m)
# mean_f1, stddev_f1 = get_mean_stddev(f1_m)
# mean_time, stddev_time = get_mean_stddev(acq_time)

# print('mean AUBC(acc): '+str(mean_acc)+'. std dev AUBC(acc): '+str(stddev_acc))
# print('mean f1: '+str(mean_f1)+'. std dev f1: '+str(stddev_f1))
# print('mean time: '+str(mean_time)+'. std dev time: '+str(stddev_time))

# file_res_tot.writelines('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc)+'\n')
# file_res_tot.writelines('mean f1: '+str(mean_f1)+'. std dev f1: '+str(stddev_f1)+'\n')
# file_res_tot.writelines('mean time: '+str(mean_time)+'. std dev acc: '+str(stddev_time)+'\n')

# # save result

# file_name_res = DATA_NAME + '_' + args_input.cell + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res.txt'
# file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')
# file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
# file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
# file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
# file_res.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
# file_res.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
# file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
# file_res.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
# file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
# avg_acc = np.mean(np.array(all_acc),axis=0)
# for i in range(len(avg_acc)):
# 	tmp = 'Size of training set is ' + str(NUM_INIT_LB + (i+1)*NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
# 	file_res.writelines(tmp)
# avg_f1 = np.mean(np.array(all_f1),axis=0)
# for i in range(len(avg_f1)):
# 	tmp = 'Size of training set is ' + str(NUM_INIT_LB + (i+1)*NUM_QUERY) + ', ' + 'f1 is ' + str(round(avg_f1[i],4)) + '.' + '\n'
# 	file_res.writelines(tmp)

# file_res.close()
# file_res_tot.close()
