import warnings
import argparse
import sys
import os
import re

import yaml
from ast import literal_eval
import copy

	
def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	#basic arguments
	parser.add_argument('--ALstrategy', '-a', default='EntropySampling', type=str, help='name of active learning strategies')
	parser.add_argument('--quota', '-q', default=30, type=int, help='quota of active learning')
	parser.add_argument('--batch', '-b', default=10, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--dataset_name', '-d', default='LINCS', type=str, help='dataset name')
	parser.add_argument('--cell', '-c', default='MCF7', type=str, help='cell line name')
	parser.add_argument('--iteration', '-t', default=5, type=int, help='time of repeat the experiment')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	parser.add_argument('--initseed', '-s', default = 0, type = int, help = 'Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default = 0, type = str, help = 'which gpu')
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	#BIM settings
	parser.add_argument('--bimeps', type=float, default=1e-3, help='learning rate of adv sample')
	parser.add_argument('--bimdis', type=float, default=0.8, help='distance threshold')
	parser.add_argument('--bimratio', type=float, default=0.7, help='ratio threshold')
	args = parser.parse_args()
	return args