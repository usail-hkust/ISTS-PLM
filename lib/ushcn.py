import os
import matplotlib
import numpy as np
import pandas as pd
import torch
import lib.utils as utils

from scipy import special
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class USHCN(object):
    """
    variables:
    "SNOW","SNWD","PRCP","TMAX","TMIN"
    """
    def __init__(self, root, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.device = device

        self.process()

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'))

        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]

    def process(self):
        if self._check_exists():
            return
        
        filename = os.path.join(self.raw_folder, 'small_chunked_sporadic.csv')
        
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing {}...'.format(filename))

        full_data = pd.read_csv(filename, index_col=0)
        full_data.index = full_data.index.astype('int32')

        entities = []
        value_cols = [c.startswith('Value') for c in full_data.columns]
        value_cols = list(full_data.columns[value_cols])
        mask_cols = [('Mask' + x[5:]) for x in value_cols]
        # print(value_cols)
        # print(mask_cols)
        data_gp = full_data.groupby(level=0) # group by index
        for record_id, data in data_gp:
            tt = torch.tensor(data['Time'].values).to(self.device).float() * (48./200)
            sorted_inds = tt.argsort() # sort over time
            vals = torch.tensor(data[value_cols].values).to(self.device).float()
            mask = torch.tensor(data[mask_cols].values).to(self.device).float()
            entities.append((record_id, tt[sorted_inds], vals[sorted_inds], mask[sorted_inds]))

        torch.save(
            entities,
            os.path.join(self.processed_folder, 'ushcn.pt')
        )

        print('Total records:', len(entities))

        print('Done!')

    def _check_exists(self):

        if not os.path.exists(os.path.join(self.processed_folder, 'ushcn.pt')):
            return False
        
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def USHCN_task_mask(args, total_dataset):
	total_dataset_new = []
	for n, (record_id, tt, vals, mask, t_bias) in enumerate(total_dataset):
		if(args.task == 'forecasting'):
			mask_observed_tp = torch.lt(tt, args.history)
		elif(args.task == 'imputation'):
			mask_observed_tp = torch.ones_like(tt).bool()
			rng = np.random.default_rng(n)
			mask_inds = rng.choice(len(tt), size=int(len(tt)*args.mask_rate), replace=False)
			
			mask_observed_tp[mask_inds] = False
		else:
			raise Exception('{}: Wrong task specified!'.format(args.task))
		total_dataset_new.append((record_id, tt, vals, mask, mask_observed_tp, t_bias))

	return total_dataset_new

def USHCN_time_chunk(data, args, device):

	chunk_data = []

	for b, (record_id, tt, vals, mask) in enumerate(data):
		for st in range(0, args.n_months - args.history - args.pred_window + 1, args.pred_window):
			et = st + args.history + args.pred_window
			if(et == args.n_months):
				indices = torch.where((tt >= st) & (tt <= et))[0]
			else:
				indices = torch.where((tt >= st) & (tt < et))[0]

			t_bias = torch.tensor(st).to(device)
			chunk_data.append((record_id, tt[indices]-t_bias, vals[indices], mask[indices], t_bias))

	return chunk_data


def USHCN_variable_time_collate_series(batch, args, device, return_np=False, to_set=False, maxlen=None,
                            data_min=None, data_max=None, time_max=None, classify=False, activity=False):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	# n_observed_tps = []
	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 
	# print(batch[0].shape)
	D = batch[0][2].shape[1]
	if maxlen is None:
		if activity == False:
			seq_lens = [ex[3].sum(dim=0).max().item() for ex in batch]
			maxlen = int(np.max(seq_lens))
		else:
			seq_lens = [ex[1].size(0) for ex in batch]
			maxlen = int(np.max(seq_lens))

	for b, (record_id, tt, vals, mask, mask_observed, t_bias) in enumerate(batch):
		
		tt = tt + t_bias
		observed_tp.append(tt[mask_observed])
		observed_data.append(vals[mask_observed])
		observed_mask.append(mask[mask_observed])
		
		mask_predicted = ~mask_observed
		predicted_tp.append(tt[mask_predicted])
		predicted_data.append(vals[mask_predicted])
		predicted_mask.append(mask[mask_predicted])
	
	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	ob_seq_lens = [ex.sum(dim=0).max().item() for ex in observed_mask]
	ob_maxlen = int(np.max(ob_seq_lens))

	# pred_seq_lens = [ex.sum(dim=0).max().item() for ex in predicted_mask]
	# pred_maxlen = int(np.max(pred_seq_lens))

	observed_combined_tt = torch.zeros([len(batch), ob_maxlen, D]).to(device)
	observed_combined_vals = torch.zeros([len(batch), ob_maxlen, D]).to(device)
	observed_combined_mask = torch.zeros([len(batch), ob_maxlen, D]).to(device)
	
	b = 0
	for tt,vals,mask in zip(observed_tp, observed_data, observed_mask):
		for d in range(D):
			mask_bd = mask[:,d].bool()
			currlen = int(mask_bd.sum())
			observed_combined_tt[b, :currlen, d] = tt[mask_bd]
			observed_combined_vals[b, :currlen, d] = vals[mask_bd,d]
			observed_combined_mask[b, :currlen, d] = mask[mask_bd,d]
		b += 1
	
	observed_tp = utils.normalize_masked_tp(observed_combined_tt, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
	# print(predicted_data.sum(), predicted_tp.sum())

	# print(observed_tp.max())
	# print(predicted_tp.max())
		
	data_dict = {"observed_data": observed_combined_vals,
			"observed_tp": observed_tp,
			"observed_mask": observed_combined_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	
	return data_dict


def USHCN_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	# n_observed_tps = []
	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 

	for b, (record_id, tt, vals, mask, mask_observed, t_bias) in enumerate(batch):

		tt = tt + t_bias
		observed_tp.append(tt[mask_observed])
		observed_data.append(vals[mask_observed])
		observed_mask.append(mask[mask_observed])
		
		mask_predicted = ~mask_observed
		predicted_tp.append(tt[mask_predicted])
		predicted_data.append(vals[mask_predicted])
		predicted_mask.append(mask[mask_predicted])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	
	observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	# print("vecdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())
	
	return data_dict

def USHCN_get_seq_length(args, records):
	
	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask, mask_observed, t_bias) in enumerate(records):
		# n_observed_tp = torch.lt(tt, args.history).sum()
		n_observed_tp = mask_observed.sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len


