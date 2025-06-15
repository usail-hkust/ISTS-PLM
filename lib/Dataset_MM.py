from math import inf
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from lib.utils import get_data_split

Constants_PAD = 0

def get_data_loader(data_x, data_y, args, shuffle=False):
    data_combined = TensorDataset(torch.from_numpy(data_x).float(),
                                        torch.from_numpy(data_y).long().squeeze())
    dataloader = DataLoader(
        data_combined, batch_size=args.batch_size, shuffle=shuffle,num_workers=8)
    
    return dataloader


def process_data(x, input_dim, m=None, tt=None, x_only=False):
    if not x_only:
        observed_vals, observed_mask, observed_tp = x[:, :,
                                                    :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
        observed_tp = np.expand_dims(observed_tp, axis=-1)
    else:
        observed_vals = x
        assert m is not None
        observed_mask = m
        observed_tp = tt

    observed_vals = tensorize_normalize(observed_vals)
    observed_vals[observed_mask == 0] = 0
    if not x_only:
        return np.concatenate((observed_vals, observed_mask, observed_tp), axis=-1)
    return observed_vals


def tensorize_normalize(P_tensor):
    mf, stdf = getStats(P_tensor)
    P_tensor = normalize(P_tensor, mf, stdf)
    return P_tensor

def getStats(P_tensor):
    N, T, F = P_tensor.shape # (B, L, D)
    Pf = P_tensor.permute(2, 0, 1).reshape(F, -1)
    mf = torch.zeros((F, 1)).to(P_tensor.device)
    stdf = torch.ones((F, 1)).to(P_tensor.device)
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if len(vals_f) > 0:
            mf[f] = torch.mean(vals_f)
            tmp_std = torch.std(vals_f)
            stdf[f] = max(tmp_std, eps)
    return mf, stdf


def normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor

def imputation(vals, mask):
    B, L, D = vals.shape
    mf, stdf = getStats(vals)
    mf_broad = mf.view(1, D, 1).repeat(B, 1, 1) # (B, D, 1)
    vals[mask==0] = torch.nan
    vals = vals.permute(0, 2, 1) # (B, L, D) -> (B, D, L)

    # foward-fill
    nan_mask = torch.isnan(vals)
    idx = torch.where(~nan_mask, torch.arange(L, device=vals.device).expand_as(vals), 0)
    idx = torch.cummax(idx, dim=-1).values 
    vals = torch.gather(vals, -1, idx) # (B, D, L)

    # back-fill
    vals = torch.flip(vals, dims=[-1])
    nan_mask = torch.isnan(vals)
    idx = torch.where(~nan_mask, torch.arange(L, device=vals.device).expand_as(vals), 0)
    idx = torch.cummax(idx, dim=-1).values 
    vals = torch.gather(vals, -1, idx) # (B, D, L)

    vals = torch.flip(vals, dims=[-1]) # (B, D, L)
    vals = torch.where(torch.isnan(vals), mf_broad, vals)

    vals = vals.permute(0, 2, 1) # (B, L, D)
    return vals

def get_data_mean_std(records, device):
    record_id, tt, vals, mask, labels = records[0]
    n_features = vals.shape[-1]
    data_records = [None for _ in range(n_features)]
    time_max = 0
    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        # tt: [L, ]
        # vals: [L, D]
        # masks: [L, D]
        # labels: scalar
        time_max = max(time_max, tt.max())
        for i in range(n_features):
            non_missing_vals = vals[:,i][mask[:,i] == 1]
            # print(non_missing_vals)
            non_missing_vals = non_missing_vals.to(device)
            if(data_records[i] == None):
                data_records[i] = non_missing_vals
            else:
                data_records[i] = torch.cat([data_records[i], non_missing_vals], dim=0)

    data_mean = torch.zeros((n_features,)).to(device)
    data_std = torch.ones((n_features,)).to(device)
    # print(data_records)
    for i in range(n_features):
        data_mean[i] = data_records[i].mean()
        data_std[i] = data_records[i].std()
    # print(data_mean, data_std, time_max)
    return data_mean, data_std, time_max


def collate_fn_triple(observed_tp, observed_data, observed_mask, device=None, data_min=None, data_max=None):
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

    batch_triple = []
    batch_mask = []
    batch_size = len(observed_data)
    for b in range(batch_size):
        norm_tt = observed_tp[b] # (L)
        norm_vals = observed_data[b] # (L, D)
        mask = observed_mask[b] # (L, D)
        norm_tt= mask * norm_tt.unsqueeze(dim=-1)
        mask_inds = torch.nonzero(mask, as_tuple=True)
        var_sel = mask_inds[1]
        tt_sel = norm_tt[mask_inds[0], mask_inds[1]]
        vals_sel = norm_vals[mask_inds[0], mask_inds[1]]
        data_triple = torch.stack([tt_sel, var_sel, vals_sel], dim=-1)
        batch_triple.append(data_triple)
        batch_mask.append(mask[mask_inds])
    
    batch_triple = pad_sequence(batch_triple, batch_first=True) # padding to front
    batch_mask = pad_sequence(batch_mask, batch_first=True)
    
    return batch_triple, batch_mask


def variable_time_collate_fn_vector(batch, args, device, input_dim, return_np=False, to_set=False, maxlen=None,
                            data_mean=None, data_std=None, time_max=None, activity=False, fillmiss=False):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: (M, T)
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    # N = batch[0][-1].shape[1] if activity else 1
    # if maxlen is None:
    #     len_tt = [ex[1].size(0) for ex in batch]
    #     maxlen = np.max(len_tt)

    if maxlen is None:
        seq_lens = [ex[1].size(0) for ex in batch]
        maxlen = int(np.max(seq_lens))

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    combined_labels = torch.zeros([len(batch)]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = min(tt.size(0),maxlen)
        enc_combined_tt[b, :currlen] = tt[:currlen].to(device)
        enc_combined_vals[b, :currlen] = vals[:currlen].to(device)
        enc_combined_mask[b, :currlen] = mask[:currlen].to(device)

        if labels.dim() == 2:
            combined_labels[b] = torch.argmax(labels,dim=-1)
        else:
            combined_labels[b] = labels.to(device)

    data_std = data_std + (data_std==0)*1e-8
    if (data_std != 0.).all():
        enc_combined_vals = (enc_combined_vals - data_mean) / data_std
        enc_combined_vals[enc_combined_mask==0] = 0
    else:
        raise Exception("Zero!")

    if time_max != 0.:
        enc_combined_tt = enc_combined_tt / time_max

    if args.fillmiss:
        enc_combined_vals = imputation(enc_combined_vals, enc_combined_mask)
    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)

    return combined_data, combined_labels, torch.tensor(seq_lens).to(device)



def variable_time_collate_fn_indseq(batch, args, device, input_dim, return_np=False, to_set=False, maxlen=None,
                            data_mean=None, data_std=None, time_max=None, activity=False, fillmiss=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: (M, T)
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[-1]
    # number of labels
    # N = batch[0][-1].shape[1] if activity else 1
    if maxlen is None:
        if activity == False:
            seq_lens = [ex[3].sum(dim=0).max().item() for ex in batch]
            maxlen = int(np.max(seq_lens))
        else:
            seq_lens = [ex[1].size(0) for ex in batch]
            maxlen = int(np.max(seq_lens))

    enc_combined_tt = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if activity:
        combined_labels = torch.zeros([len(batch), maxlen]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        # currlen = min(int(mask.sum(0).max()),maxlen)
        for d in range(D):
            mask_bd = mask[:,d].bool()
            currlen = int(mask_bd.sum())
            enc_combined_tt[b, :currlen, d] = tt[mask_bd]
            enc_combined_vals[b, :currlen, d] = vals[mask_bd,d]
            enc_combined_mask[b, :currlen, d] = mask[mask_bd,d]
        if labels.dim() == 2:
            print(record_id)
            combined_labels[b] = torch.argmax(labels,dim=-1)
        else:
            combined_labels[b] = labels.to(device)

    data_std = data_std + (data_std==0)*1e-8
    if (data_std != 0.).all():
        enc_combined_vals = (enc_combined_vals - data_mean) / data_std
        enc_combined_vals[enc_combined_mask==0] = 0
    else:
        raise Exception("Zero!")

    if time_max != 0.:
        enc_combined_tt = enc_combined_tt / time_max

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt), 2)
    
    return combined_data, combined_labels, torch.tensor(seq_lens).to(device)

def get_time_PAM(data):
    T,F = data[0].shape
    time = np.zeros((len(data), T, 1))
    for i in range(len(data)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        time[i] = tim
    time = torch.Tensor(time) / 60.0
    time = time.squeeze(-1)
    return time

def get_PAM_data(args, device):
    split_path='/splits/PAMAP2_split_'+args.split+'.npy'
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path='./data/PAMAP2data',
                                                              split_path=split_path,
                                                              dataset='PAM'
                                                              )
    T, F = Ptrain[0].shape
    train_time = get_time_PAM(Ptrain)
    val_time = get_time_PAM(Pval)
    test_time = get_time_PAM(Ptest)
    Ptrain = torch.Tensor(Ptrain)
    Pval = torch.Tensor(Pval)
    Ptest = torch.Tensor(Ptest)
    ytrain = torch.Tensor(ytrain)
    yval = torch.Tensor(yval)
    ytest = torch.Tensor(ytest)
    train_mask = torch.zeros_like(Ptrain)
    train_mask[Ptrain > 0] = 1
    val_mask = torch.zeros_like(Pval)
    val_mask[Pval > 0] = 1
    test_mask = torch.zeros_like(Ptest)
    test_mask[Ptest > 0] = 1
    train_data = []
    b = 0
    for data,tt,mask in zip(Ptrain, train_time, train_mask):
        train_data.append((b, tt, data, mask, ytrain[b]))
        b += 1
    val_data = []
    b = 0
    for data,tt,mask in zip(Pval, val_time, val_mask):
        val_data.append((b, tt, data, mask, yval[b]))
        b += 1
    test_data = []
    b = 0
    for data,tt,mask in zip(Ptest, test_time, test_mask):
        test_data.append((b, tt, data, mask, ytest[b]))
        b += 1

    seen_data = train_data + val_data
    if args.few_shot:
        train_data, _ = model_selection.train_test_split(train_data, train_size=0.1, random_state=42, shuffle=True)
        print(train_data[0])
        print('--few_shot--', len(train_data))
    test_record_ids = [record_id for record_id, tt, vals, mask, label in test_data]
    
    print("train, val, test data split:", len(train_data), len(val_data), len(test_data))
    print("Test record ids (first 20):", test_record_ids[:20])
    print("Test record ids (last 20):", test_record_ids[-20:])
    
    record_id, tt, vals, mask, labels = train_data[0]
    data_mean, data_std, time_max = get_data_mean_std(seen_data, device)
    print("data norm:", data_mean.sum(), data_std.sum())

    input_dim = vals.size(-1)
    batch_size = min(len(seen_data), args.batch_size)
    args.num_types = input_dim

    if args.collate == 'indseq':
        collate_fn = variable_time_collate_fn_indseq
    
    else:
        collate_fn = variable_time_collate_fn_vector

    max_len = -1


    train_data_combined = collate_fn(train_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, train_data_combined[0].shape[1])
    val_data_combined = collate_fn(val_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, val_data_combined[0].shape[1])
    test_data_combined = collate_fn(test_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, test_data_combined[0].shape[1])

    print("collate data shape:", train_data_combined[0].shape, val_data_combined[0].shape, test_data_combined[0].shape)
    print("collate label shape:", train_data_combined[1].shape, val_data_combined[1].shape, test_data_combined[1].shape)

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze(), train_data_combined[2])
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long().squeeze(), val_data_combined[2])
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze(), test_data_combined[2])

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, input_dim, max_len

def get_processed_data(data, label):
    new_data = []
    for patient,tag in zip(data, label):
        id = patient['id']
        time = torch.Tensor(patient['time'])
        for i in range(1, len(time)):
            if time[i] == 0:
                first_zero_index_after_first_element = i
                break
        time = time[:first_zero_index_after_first_element]
        time = time.squeeze(-1)
        arr = torch.Tensor(patient['arr'])
        arr = arr[:first_zero_index_after_first_element]
        mask = torch.zeros_like(arr)
        mask[arr>0] = 1
        label_mor = torch.Tensor([tag[-1]])
        new_data.append((id, time, arr, mask, label_mor))
    return new_data

def get_P12_data(args, device):

    split_path='/splits/phy12_split'+args.split+'.npy'
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path='./data/P12data',
                                                              split_path=split_path,
                                                              dataset='P12',
                                                              debug=args.debug
                                                              )
    train_data = get_processed_data(Ptrain, ytrain)
    val_data = get_processed_data(Pval, yval)
    test_data = get_processed_data(Ptest, ytest)

    seen_data = train_data + val_data
    if args.few_shot:
        train_data, _ = model_selection.train_test_split(train_data, train_size=0.1, random_state=42, shuffle=True)
        print(train_data[0])
        print('--few_shot--', len(train_data))
    test_record_ids = [record_id for record_id, tt, vals, mask, label in test_data]
    
    print("train, val, test data split:", len(train_data), len(val_data), len(test_data))
    print("Test record ids (first 20):", test_record_ids[:20])
    print("Test record ids (last 20):", test_record_ids[-20:])
    
    record_id, tt, vals, mask, labels = train_data[0]
    data_mean, data_std, time_max = get_data_mean_std(seen_data, device)
    print("data norm:", data_mean.sum(), data_std.sum())

    input_dim = vals.size(-1)
    batch_size = min(len(seen_data), args.batch_size)
    args.num_types = input_dim

    if args.collate == 'indseq':
        collate_fn = variable_time_collate_fn_indseq
    
    else:
        collate_fn = variable_time_collate_fn_vector

    max_len = -1
    train_data_combined = collate_fn(train_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, train_data_combined[0].shape[1])
    val_data_combined = collate_fn(val_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, val_data_combined[0].shape[1])
    test_data_combined = collate_fn(test_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, test_data_combined[0].shape[1])


    print("collate data shape:", train_data_combined[0].shape, val_data_combined[0].shape, test_data_combined[0].shape)
    print("collate label shape:", train_data_combined[1].shape, val_data_combined[1].shape, test_data_combined[1].shape)

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze(), train_data_combined[2])
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long().squeeze(), val_data_combined[2])
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze(), test_data_combined[2])

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, input_dim, max_len

def get_P19_data(args, device):
    split_path='/splits/phy19_split'+args.split+'_new.npy'
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path='./data/P19data',
                                                              split_path=split_path,
                                                              dataset='P19',
                                                              debug=args.debug
                                                              )
    train_data = get_processed_data(Ptrain, ytrain)
    val_data = get_processed_data(Pval, yval)
    test_data = get_processed_data(Ptest, ytest)

    seen_data = train_data + val_data
    if args.few_shot:
        train_data, _ = model_selection.train_test_split(train_data, train_size=0.1, random_state=42, shuffle=True)
        print(train_data[0])
        print('--few_shot--', len(train_data))
    test_record_ids = [record_id for record_id, tt, vals, mask, label in test_data]
    
    print("train, val, test data split:", len(train_data), len(val_data), len(test_data))
    print("Test record ids (first 20):", test_record_ids[:20])
    print("Test record ids (last 20):", test_record_ids[-20:])
    
    record_id, tt, vals, mask, labels = train_data[0]
    data_mean, data_std, time_max = get_data_mean_std(seen_data, device)
    print("data norm:", data_mean.sum(), data_std.sum())

    input_dim = vals.size(-1)
    batch_size = min(len(seen_data), args.batch_size)
    args.num_types = input_dim

    if args.collate == 'indseq':
        collate_fn = variable_time_collate_fn_indseq
    
    else:
        collate_fn = variable_time_collate_fn_vector

    max_len = -1

    train_data_combined = collate_fn(train_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, train_data_combined[0].shape[1])
    val_data_combined = collate_fn(val_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, val_data_combined[0].shape[1])
    test_data_combined = collate_fn(test_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max)
    max_len = max(max_len, test_data_combined[0].shape[1])

    print("collate data shape:", train_data_combined[0].shape, val_data_combined[0].shape, test_data_combined[0].shape)
    print("collate label shape:", train_data_combined[1].shape, val_data_combined[1].shape, test_data_combined[1].shape)

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze(), train_data_combined[2])
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long().squeeze(), val_data_combined[2])
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze(), test_data_combined[2])

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, input_dim, max_len

def get_processed_data_static(data, label, use_static=False):
    new_data = []
    static_list = []
    for patient,tag in zip(data, label):
        if use_static:
            static_vars = torch.Tensor(patient['extended_static'])
            len_static = len(static_vars)
            # static = torch.Tensor(patient['static'])
        id = patient['id']
        time = torch.Tensor(patient['time'])
        for i in range(1, len(time)):
            if time[i] == 0:
                first_zero_index_after_first_element = i
                break
        time = time[:first_zero_index_after_first_element]
        time = time.squeeze(-1)
        # if(time[0] != 0):
        #     time = torch.cat([torch.Tensor([0.0]), time])
        time_len = time.size(0)
        arr = torch.Tensor(patient['arr'])
        arr = arr[:first_zero_index_after_first_element]

        if use_static:
            # concat static features
            static_pad = torch.zeros([arr.shape[0],len_static])
            static_pad[0] = static_vars
            arr = torch.cat([arr, static_pad], dim=-1)
        
        mask = torch.zeros_like(arr)
        mask[arr>0] = 1
        if use_static:
            mask[0,-len_static:] = 1
        label_mor = torch.Tensor([tag[-1]])
        # print(time.shape, arr.shape, mask.shape)
        new_data.append((id, time, arr, mask, label_mor))
        static_list.append(torch.Tensor(patient['static']))

    return new_data, static_list

def get_P12_data_zeroshot(args, device):

    split_path='/splits/phy12_split'+args.split+'.npy'
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path='./data/P12data',
                                                              split_path=split_path,
                                                              dataset='P12',
                                                              debug=args.debug
                                                              )
    train_data, train_static = get_processed_data_static(Ptrain, ytrain)
    val_data, val_static = get_processed_data_static(Pval, yval)
    test_data, test_static = get_processed_data_static(Ptest, ytest)

    all_data = train_data + val_data + test_data
    all_static = train_static + val_static + test_static
    seen_data = []
    test_data = []
    if args.zero_shot_ICU:
        for (patient, static) in zip(all_data, all_static):
            if int(static[-2]) == 3:
                test_data.append(patient)
            else:
                seen_data.append(patient)
    elif args.zero_shot_age:
        for (patient, static) in zip(all_data, all_static):
            if int(static[0]) < 65:
                test_data.append(patient)
            else:
                seen_data.append(patient)
    print('seen_data', len(seen_data))
    print('test_data', len(test_data))
    train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.8, random_state=42, shuffle=True)
    test_record_ids = [record_id for record_id, tt, vals, mask, label in test_data]
    
    print("train, val, test data split:", len(train_data), len(val_data), len(test_data))
    print("Test record ids (first 20):", test_record_ids[:20])
    print("Test record ids (last 20):", test_record_ids[-20:])
    
    record_id, tt, vals, mask, labels = train_data[0]
    data_mean, data_std, time_max = get_data_mean_std(seen_data, device)
    print("data norm:", data_mean.sum(), data_std.sum())

    input_dim = vals.size(-1)
    batch_size = min(len(seen_data), args.batch_size)
    args.num_types = input_dim

    if args.collate == 'indseq':
        collate_fn = variable_time_collate_fn_indseq
    
    else:
        collate_fn = variable_time_collate_fn_vector

    max_len = -1
    train_data_combined = collate_fn(train_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, train_data_combined[0].shape[1])
    val_data_combined = collate_fn(val_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, val_data_combined[0].shape[1])
    test_data_combined = collate_fn(test_data, args=args, device=device,\
                        input_dim=input_dim, data_mean=data_mean, data_std=data_std, time_max=time_max, fillmiss=args.fillmiss)
    max_len = max(max_len, test_data_combined[0].shape[1])


    print("collate data shape:", train_data_combined[0].shape, val_data_combined[0].shape, test_data_combined[0].shape)
    print("collate label shape:", train_data_combined[1].shape, val_data_combined[1].shape, test_data_combined[1].shape)

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze(), train_data_combined[2])
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long().squeeze(), val_data_combined[2])
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze(), test_data_combined[2])

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, input_dim, max_len