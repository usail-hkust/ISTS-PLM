import numpy as np
import pandas as pd
import os
import math
import sys
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import time
import random
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None,epoch=None):

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # gpu

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