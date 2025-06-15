
import os
import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math 
import glob
import re
import sys
from shutil import copyfile
import subprocess
import datetime
import time
import random
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from einops import rearrange, repeat


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
    

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False, mode='a'):
	logger = logging.getLogger(logpath)
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath)
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger



def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def make_dataset(dataset_type = "spiral",**kwargs):
	if dataset_type == "spiral":
		data_path = "data/spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	elif dataset_type == "chiralspiral":
		data_path = "data/chiral-spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	else:
		raise Exception("Unknown dataset type " + dataset_type)
	return dataset, chiralities


def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res


def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)


def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))


def subsample_timepoints(data, time_steps, mask, n_tp_to_sample = None):
	# n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
	if n_tp_to_sample is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)


	if n_tp_to_sample > 1:
		# Subsample exact number of points
		assert(n_tp_to_sample <= n_tp_in_batch)
		n_tp_to_sample = int(n_tp_to_sample)

		for i in range(data.size(0)):
			missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

			data[i, missing_idx] = 0.
			if mask is not None:
				mask[i, missing_idx] = 0.
	
	elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
		# Subsample percentage of points from each time series
		percentage_tp_to_sample = n_tp_to_sample
		for i in range(data.size(0)):
			# take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
			current_mask = mask[i].sum(-1).cpu()
			non_missing_tp = np.where(current_mask > 0)[0]
			n_tp_current = len(non_missing_tp)
			n_to_sample = int(n_tp_current * percentage_tp_to_sample)
			subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
			tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

			data[i, tp_to_set_to_zero] = 0.
			if mask is not None:
				mask[i, tp_to_set_to_zero] = 0.

	return data, time_steps, mask



def cut_out_timepoints(data, time_steps, mask, n_points_to_cut = None):
	# n_points_to_cut: number of consecutive time points to cut out
	if n_points_to_cut is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)

	if n_points_to_cut < 1:
		raise Exception("Number of time points to cut out must be > 1")

	assert(n_points_to_cut <= n_tp_in_batch)
	n_points_to_cut = int(n_points_to_cut)

	for i in range(data.size(0)):
		start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)

		data[i, start : (start + n_points_to_cut)] = 0.
		if mask is not None:
			mask[i, start : (start + n_points_to_cut)] = 0.

	return data, time_steps, mask

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()


def split_train_test(data, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]
	return data_train, data_test

def split_train_test_data_and_time(data, time_steps, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]

	assert(len(time_steps.size()) == 2)
	train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
	test_time_steps = time_steps[:, int(n_samples * train_fraq):]

	return data_train, data_test, train_time_steps, test_time_steps

def get_next_batch(dataloader):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()
	
	return data_dict

def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path)
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)


def get_item_from_pickle(pickle_file, item_name):
	from_pickle = load_pickle(pickle_file)
	if item_name in from_pickle:
		return from_pickle[item_name]
	return None


def get_dict_template():
	return {"observed_data": None,
			"observed_tp": None,
			"data_to_predict": None,
			"tp_to_predict": None,
			"observed_mask": None,
			"mask_predicted_data": None,
			# "labels": None
			}

def normalize_data(data):
	reshaped = data.reshape(-1, data.size(-1))

	att_min = torch.min(reshaped, 0)[0]
	att_max = torch.max(reshaped, 0)[0]
	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / att_max
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
	# data = torch.clamp(data, min=att_min, max=att_max) * mask
	# scale = att_max
	scale = att_max - att_min
	scale = scale + (scale == 0) * 1e-8
	# we don't want to divide by zero
	if (scale != 0.).all(): 
		data_norm = (data - att_min) / scale
	else:
		raise Exception("Zero!")

	# set masked out elements back to zero 
	data_norm[mask == 0] = 0

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm

def normalize_masked_tp(data, att_min, att_max):
	scale = att_max - att_min
	scale = scale + (scale == 0) * 1e-8
	# we don't want to divide by zero
	if (scale != 0.).all():
		data_norm = (data - att_min) / scale
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm


def shift_outputs(outputs, first_datapoint = None):
	outputs = outputs[:,:,:-1,:]

	if first_datapoint is not None:
		n_traj, n_dims = first_datapoint.size()
		first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
		outputs = torch.cat((first_datapoint, outputs), 2)
	return outputs

def split_and_patch_batch(data_dict, args, patch_indices):

	device = get_device(data_dict["data"])

	split_dict = {"tp_to_predict": data_dict["tp_to_predict"].clone(),
			"data_to_predict": data_dict["data_to_predict"].clone(),
			"mask_predicted_data": data_dict["mask_predicted_data"].clone()
			}
	
	observed_tp = data_dict["time_steps"].clone() # (n_observed_tp, )
	observed_data = data_dict["data"].clone() # (bs, n_observed_tp, D)
	observed_mask = data_dict["mask"].clone() # (bs, n_observed_tp, D)

	n_batch, n_tp, n_dim = observed_data.shape
	observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(n_batch, args.npatch, 1, n_dim)
	observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)
	observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)

	max_patch_len = 0
	for i in range(args.npatch):
		indices = patch_indices[i]
		if(len(indices) == 0): continue
		st_ind, ed_ind = indices[0], indices[-1]
		n_data_points = observed_mask[:, st_ind:ed_ind+1].sum(dim=1).max().item()
		max_patch_len = max(max_patch_len, int(n_data_points))

	observed_mask_patches_fill = torch.zeros_like(observed_mask_patches, dtype=observed_mask.dtype) # n_batch, npacth, n_tp, n_dim
	patch_indices_fianl = torch.full((n_batch, args.npatch, max_patch_len, n_dim), n_tp).to(device) # n_batch, npacth, max_patch_len, n_dim
	observed_mask_patches_fill_reindex = torch.zeros_like(patch_indices_fianl, dtype=observed_mask.dtype)
	aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)
	for i in range(args.npatch):
		indices = patch_indices[i]
		if(len(indices) == 0): continue
		st_ind, ed_ind = indices[0], indices[-1]
		observed_mask_patches_fill[:, i, st_ind:ed_ind+1] = observed_mask[:, st_ind:ed_ind+1, :]
		L = observed_mask[:, st_ind:ed_ind+1, :].sum(dim=1, keepdim=True) # (bs, 1, D)
		observed_mask_patches_fill_reindex[:, i] = (aux_tensor < L)  # let first L[i] to be True
	
	### return a indices tuple like ([...], [...], [...], [...])
	mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0,1,3,2), as_tuple=True) # reset indices
	ind_values = torch.nonzero(observed_mask_patches_fill.permute(0,1,3,2), as_tuple=True)[-1] # original indices of dimension 2

	### fill n_tp if the number of observed points are less than max_patch_len
	patch_indices_fianl.index_put_((mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values)

	pad_zeros_data = torch.zeros([n_batch, args.npatch, 1, n_dim]).to(device)
	observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl) # (n_batch, npatch, max_patch_len, n_dim)
	observed_data_patches = torch.cat([observed_data_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)
	observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)

	split_dict["observed_tp"] = observed_tp_patches
	split_dict["observed_data"] = observed_data_patches
	split_dict["observed_mask"] = observed_mask_patches 

	return split_dict

### version 2
def split_data_forecast(data_dict, dataset, n_observed_tp):

	device = get_device(data_dict["data"])
	split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
				"observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
				"data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
				"tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone()}

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if ("mask" in data_dict) and (data_dict["mask"] is not None):
		split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()
	split_dict["mode"] = "forecast"

	return split_dict

def split_data_interp(data_dict):
	device = get_device(data_dict["data"])

	split_dict = {"observed_data": data_dict["data"].clone(),
				"observed_tp": data_dict["time_steps"].clone(),
				"data_to_predict": data_dict["data"].clone(),
				"tp_to_predict": data_dict["time_steps"].clone()}

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if "mask" in data_dict and data_dict["mask"] is not None:
		split_dict["observed_mask"] = data_dict["mask"].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"].clone()

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		split_dict["labels"] = data_dict["labels"].clone()

	split_dict["mode"] = "interp"
	return split_dict

def add_mask(data_dict):
	data = data_dict["observed_data"]
	mask = data_dict["observed_mask"]

	if mask is None:
		mask = torch.ones_like(data).to(get_device(data))

	data_dict["observed_mask"] = mask
	return data_dict


def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
	# n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
	# n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points
	if n_tp_to_sample is not None:
		# Randomly subsample time points
		data, time_steps, mask = subsample_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_tp_to_sample = n_tp_to_sample)

	if n_points_to_cut is not None:
		# Remove consecutive time points
		data, time_steps, mask = cut_out_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_points_to_cut = n_points_to_cut)

	new_data_dict = {}
	for key in data_dict.keys():
		new_data_dict[key] = data_dict[key]

	new_data_dict["observed_data"] = data.clone()
	new_data_dict["observed_tp"] = time_steps.clone()
	new_data_dict["observed_mask"] = mask.clone()

	if n_points_to_cut is not None:
		# Cut the section in the data to predict as well
		# Used only for the demo on the periodic function
		new_data_dict["data_to_predict"] = data.clone()
		new_data_dict["tp_to_predict"] = time_steps.clone()
		new_data_dict["mask_predicted_data"] = mask.clone()

	return new_data_dict


def split_and_subsample_batch(data_dict, args, n_observed_tp):

	processed_dict = split_data_forecast(data_dict, args.dataset, n_observed_tp)
	# add mask
	processed_dict = add_mask(processed_dict)
	# Subsample points or cut out the whole section of the timeline
	return processed_dict


def compute_loss_all_batches(model,
	test_dataloader, args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1., 
	max_samples_for_eval = None):

	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["mse"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	total["pois_likelihood"] = 0
	total["ce_loss"] = 0

	n_test_samples = 0
	
	classif_predictions = torch.Tensor([]).to(device)
	all_test_labels =  torch.Tensor([]).to(device)

	for i in range(n_batches):
		# print("Computing loss... " + str(i))
		
		batch_dict = get_next_batch(test_dataloader)
		bs = batch_dict["observed_data"].shape[0]

		results  = model.compute_all_losses(batch_dict,
			n_traj_samples = n_traj_samples, kl_coef = kl_coef)

		if args.classif:
			n_labels = model.n_labels #batch_dict["labels"].size(-1)
			n_traj_samples = results["label_predictions"].size(0)

			classif_predictions = torch.cat((classif_predictions, 
				results["label_predictions"].reshape(n_traj_samples, -1, n_labels)),1)
			all_test_labels = torch.cat((all_test_labels, 
				batch_dict["labels"].reshape(-1, n_labels)),0)

		for key in total.keys(): 
			if key in results:
				var = results[key]
				if isinstance(var, torch.Tensor):
					var = var.detach()
				total[key] += var*bs

		n_test_samples += bs

	if n_test_samples > 0:
		for key, value in total.items():
			total[key] = total[key] / n_test_samples
	return total

def check_mask(data, mask):
	#check that "mask" argument indeed contains a mask for data
	n_zeros = torch.sum(mask == 0.).cpu().numpy()
	n_ones = torch.sum(mask == 1.).cpu().numpy()

	# mask should contain only zeros and ones
	assert((n_zeros + n_ones) == np.prod(list(mask.size())))

	# all masked out elements should be zeros
	assert(torch.sum(data[mask == 0.] != 0.) == 0)



# Classfication task
def pred_loss(prediction, truth, loss_func):
    """ supervised prediction loss, cross entropy or label smoothing. 
    prediction: [B, 2]
    label: [B]
    """
    loss = loss_func(prediction, truth)
    loss = torch.sum(loss)
    return loss

def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]

	if len(pred_y.shape) == 3: 
		pred_y = pred_y.unsqueeze(dim=0)
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1, n_dim).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1, n_dim).sum(dim=0) # (n_dim, )

	if(reduce == "mean"):
		### 1. Compute avg error of each variable first 
		### 2. Compute avg error along the variables 
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / n_avai_var # (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")

def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        classifier_state_dict = None

        if dp_flag:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if classifier is not None:
            classifier_state_dict = classifier.state_dict()
            
        if self.save_path is not None:
            torch.save({
                'model_state_dict':model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, self.save_path)
        else:
            print("no path assigned")  

        self.val_loss_min = val_loss


def log_info(opt, phase, epoch, acc, rmse=0.0, start=0.0, value_rmse=0.0, auroc=0.0, auprc=0.0, loss=0.0, precision=0.0, recall=0.0, F1=0.0, save=False):
    if opt.task == 'PAM':
        print('  -(', phase, ') epoch: {epoch}, acc: {acc: 8.5f}, auroc: {auroc: 8.5f}, auprc: {auprc: 8.5f}, '
                    'precision: {precision: 8.5f}, recall: {recall: 8.5f}, F1: {F1: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                    .format(epoch=epoch, acc=acc,  auroc=auroc, auprc=auprc, precision=precision, recall=recall, F1=F1, loss=loss, elapse=(time.time() - start) / 60))

        if save and opt.log is not None:
            with open(opt.log, 'a') as f:
                f.write(phase + ':\t{epoch},  ACC: {acc: 8.5f}, auroc: {auroc: 8.5f}, auprc: {auprc: 8.5f}, precision: {precision: 8.5f}, recall: {recall: 8.5f}, F1: {F1: 8.5f}, Loss: {loss: 8.5f}\n'
                        .format(epoch=epoch, acc=acc, auroc=auroc, auprc=auprc, precision=precision, recall=recall, F1=F1, loss=loss))
    else:
        print('  -(', phase, ') epoch: {epoch}, RMSE: {rmse: 8.5f}, acc: {type: 8.5f}, '
                    'AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, Value_RMSE: {value_rmse: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                    .format(epoch=epoch, type=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss, elapse=(time.time() - start) / 60))

        if save and opt.log is not None:
            with open(opt.log, 'a') as f:
                f.write(phase + ':\t{epoch}, TimeRMSE: {rmse: 8.5f},  ACC: {acc: 8.5f}, AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, ValueRMSE: {value_rmse: 8.5f}, Loss: {loss: 8.5f}\n'
                        .format(epoch=epoch, acc=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss))

def log_info_forecast(opt, phase, epoch, start=0.0, mse=0.0, mae=0.0, loss=0.0, save=False):
    print('  -(', phase, ') epoch: {epoch}, MSE: {mse: .6f}, MAE: {mae: .6f}, '
                'loss: {loss: .6f}, elapse: {elapse:3.3f} min'
                .format(epoch=epoch, mse=mse, mae=mae, loss=loss, elapse=(time.time() - start) / 60))

    if save and opt.log is not None:
        with open(opt.log, 'a') as f:
            f.write(phase + ':\t{epoch}, MSE: {mse: .6f}, MAE: {mae: .6f}, Loss: {loss: .6f}\n'
                    .format(epoch=epoch, mse=mse, mae=mae, loss=loss))               

def load_checkpoints(save_path, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False, use_cpu=False):
    if not os.path.getsize(save_path) > 0: 
        print(save_path, " is None file")
        sys.exit(0)

    if use_cpu:
        checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(save_path)
    
    if dp_flag:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])


    if classifier is not None and checkpoint['classifier_state_dict'] is not None:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

    if time_predictor is not None and checkpoint['time_predictor_state_dict'] is not None:
        time_predictor.load_state_dict(checkpoint['time_predictor_state_dict'])

    if decoder is not None and checkpoint['decoder_state_dict'] is not None:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    return model, classifier, time_predictor, decoder

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]   

def evaluate_mc(label, pred, n_class):
    ypred = np.argmax(pred, axis=1)
    denoms = np.sum(np.exp(pred), axis=1).reshape((-1, 1))
    probs = np.exp(pred) / denoms

    if n_class == 2:
        acc = np.sum(label.ravel() == ypred.ravel()) / pred.shape[0]
        precision = metrics.precision_score(label, ypred)
        recall = metrics.recall_score(label, ypred)
        F1 = metrics.f1_score(label, ypred)
        auroc = metrics.roc_auc_score(label, probs[:, 1])
        auprc = metrics.average_precision_score(label, probs[:, 1])
    elif n_class > 2:
        acc = np.sum(label.ravel() == ypred.ravel()) / pred.shape[0]
        precision = metrics.precision_score(label, ypred, average="macro")
        recall = metrics.recall_score(label, ypred, average="macro") 
        F1 = metrics.f1_score(label, ypred, average="macro")
        auroc = metrics.roc_auc_score(one_hot(label), probs)
        auprc = metrics.average_precision_score(one_hot(label), probs)

    return acc, auroc, auprc, precision, recall, F1

def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True, dataset='PAM', predictive_label='mortality', debug=False):
    # load data
    if dataset == 'P12':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)

    if split_type == 'random':
        # load random indices from a split
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]
    print("#Train-val-test:", len(Ptrain), len(Pval), len(Ptest))

    y = arr_outcomes[:, -1].reshape((-1, 1))

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]
    if(debug):
        ytrain = ytrain[:60]
        yval = yval[:20]
        ytest = ytest[:20]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest