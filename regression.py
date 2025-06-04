import os
import sys
sys.path.append("..")

import time
import argparse
import numpy as np
import pandas as pd
import datetime
from random import SystemRandom
from models.gpt4ts import *

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils

from lib.parse_datasets import parse_datasets
from lib.evaluation import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('ITS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('--model', type=str, default='gpt', help='select from [gpt, gpt_patch, warpformer]')

parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='../data/')

parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_types', type=int, default=23)

parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--log', type=str, default='./logs/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--task', type=str, default='nan')

parser.add_argument('--debug_flag', action='store_true')
parser.add_argument('--dp_flag', action='store_true')
parser.add_argument('--load_in_batch', action='store_true')
parser.add_argument('--history', type=int, default=24, help="number of hours (or months for ushcn) as historical window")

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--median_len', type=int, default=50)
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
### gpt4ts
parser.add_argument('--n_te_gptlayer', type=int, default=6)
parser.add_argument('--n_st_gptlayer', type=int, default=6)
parser.add_argument('--te_model', type=str, default='gpt')
parser.add_argument('--st_model', type=str, default='bert')
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--semi_freeze', action='store_true')
parser.add_argument('--sample_rate', type=float, default=1.0)
parser.add_argument('--mask_rate', type=float, default=0.3)
parser.add_argument('--collate', type=str, default='indseq')


args = parser.parse_args()
file_name = os.path.basename(__file__)[:-3]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################


if __name__ == '__main__':

	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	# if not os.path.exists(args.save):
	# 	utils.makedirs(args.save)
	# ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)
	

	if(args.n < 12000):
		args.state = "debug"
		log_path = f"logs/{args.task}_{args.dataset}_{args.model}_{args.state}.log"
	else:
		log_path = f"logs/{args.task}_{args.dataset}_{args.model}_{args.state}_maskrate{args.mask_rate}_history{args.history}\
			_plmlayer{args.n_te_gptlayer}_{args.n_st_gptlayer}_lr{args.lr}.log"
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")

	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	##################################################################
	data_obj = parse_datasets(args, length_stat=True)
	args.enc_in = data_obj["input_dim"]
	args.num_types =  data_obj["input_dim"]
	args.input_dim =  data_obj["input_dim"]
	args.median_len = data_obj["median_len"]
	args.input_len = data_obj["max_input_len"]
	args.pred_len = data_obj["max_pred_len"]
	
	### Model Config ###
	if(args.model == 'istsplm_forecast'):
		model = istsplm_forecast(args).to(args.device)
	elif(args.model == 'istsplm_vector_forecast'):
		model = istsplm_vector_forecast(args).to(args.device)
	elif(args.model == 'istsplm_set_forecast'):
		model = istsplm_set_forecast(args).to(args.device)
	
	### Optimizer ###
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			# utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			train_res = compute_all_losses(model, batch_dict)
			train_res["loss"].backward()
			optimizer.step()

		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			if(val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			
			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
			 		 test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))

		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			sys.exit(0)
