
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import lib.utils as utils
from lib.generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.person_activity import *
from lib.mimic import *
from lib.ushcn import *
from sklearn import model_selection

#####################################################################################################

def task_mask(args, total_dataset):
	total_dataset_new = []
	for n, (record_id, tt, vals, mask) in enumerate(total_dataset):
		if(args.task == 'forecasting'):
			mask_observed_tp = torch.lt(tt, args.history)
		elif(args.task == 'imputation'):
			mask_observed_tp = torch.ones_like(tt).bool()
			rng = np.random.default_rng(n)
			mask_inds = rng.choice(len(tt), size=int(len(tt)*args.mask_rate), replace=False)
			mask_observed_tp[mask_inds] = False
			if args.dataset in ["physionet"]:
				mask_observed_tp[0] = True # in some datasets, only time 0 has values 
		else:
			raise Exception('{}: Wrong task specified!'.format(args.task))
		total_dataset_new.append((record_id, tt, vals, mask, mask_observed_tp))

	return total_dataset_new

def parse_datasets(args, length_stat=False):

	device = args.device
	dataset_name = args.dataset

	##################################################################
	### PhysioNet&Mimic dataset ### 
	if dataset_name in ["physionet", "mimic"]:

		### list of tuples (record_id, tt, vals, mask) ###
		if dataset_name == "physionet":
			total_dataset = PhysioNet('./data/physionet', quantization = args.quantization,
											download=True, n_samples = args.n, device = device)
		elif dataset_name == "mimic":
			total_dataset = MIMIC('./data/mimic/', n_samples = args.n, device = device)

		total_dataset = task_mask(args, total_dataset)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask, mask_observed in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask, mask_observed = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

		if(args.collate =='indseq'):
			collate_fn = variable_time_collate_series
		else:
			collate_fn = variable_time_collate_fn

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects		

	##################################################################
	### Activity dataset ###
	elif dataset_name == "activity":
		args.pred_window = 1000 # predict future 1000 ms

		total_dataset = PersonActivity('./data/activity/', n_samples = args.n, download=True, device = device)
		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, _ = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)
		time_max = torch.tensor(args.history + args.pred_window)
		print('manual set time_max:', time_max)

		if(args.collate =='indseq'):
			collate_fn = variable_time_collate_series
		else:
			collate_fn = variable_time_collate_fn

		train_data = Activity_time_chunk(train_data, args, device)
		train_data = task_mask(args, train_data)
		val_data = Activity_time_chunk(val_data, args, device)
		val_data = task_mask(args, val_data)
		test_data = Activity_time_chunk(test_data, args, device)
		test_data = task_mask(args, test_data)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device,
				data_min = data_min, data_max = data_max, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
	
	##################################################################
	### USHCN dataset ###
	elif dataset_name == "ushcn":
		args.n_months = 48 # 48 monthes
		args.pred_window = 1 # predict future one month

		### list of tuples (record_id, tt, vals, mask) ###
		total_dataset = USHCN('./data/ushcn/', n_samples = args.n, device = device)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.6, random_state = 42, shuffle = False)

		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

		if(args.collate =='indseq'):
			collate_fn = USHCN_variable_time_collate_series
		else:
			collate_fn = USHCN_variable_time_collate_fn

		train_data = USHCN_time_chunk(train_data, args, device)
		train_data = USHCN_task_mask(args, train_data)
		val_data = USHCN_time_chunk(val_data, args, device)
		val_data = USHCN_task_mask(args, val_data)
		test_data = USHCN_time_chunk(test_data, args, device)
		test_data = USHCN_task_mask(args, test_data)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
	