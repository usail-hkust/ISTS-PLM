import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import gc
from tqdm import tqdm
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from lib.utils import *
from lib.Dataset_MM import get_PAM_data, get_P12_data, get_P19_data, get_P12_data_zeroshot
from models.plm4ts import *

eps=1e-7

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler):
    """ Epoch operation in training phase. """

    model.train()
    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0,0,0

    training_data_list = list(training_data)
    num_total_batches = len(training_data_list)
    num_sampled_batches = int(num_total_batches * opt.sample_rate)

    sampled_indices = np.random.choice(num_total_batches, size=num_sampled_batches, replace=False)

    sampled_training_data = [training_data_list[i] for i in sampled_indices]

    for train_batch in tqdm(sampled_training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        train_batch, labels, seq_lens = map(lambda x: x.to(opt.device), train_batch)
        max_len = int(seq_lens.max().item())
        observed_data, observed_mask, observed_tp = \
            train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2*opt.num_types],\
                  train_batch[:, :max_len, 2*opt.num_types:3*opt.num_types]
        del train_batch

        """ forward """
        optimizer.zero_grad()

        out = model(observed_tp, observed_data, observed_mask, opt) # [B,D]
        sup_pred = classifier(out)
        
        if sup_pred.dim() == 1:
            sup_pred = sup_pred.unsqueeze(0)

        loss = torch.sum(pred_loss_func((sup_pred), labels))
        # sup_pred = torch.softmax(sup_pred, dim=-1)

        if torch.any(torch.isnan(loss)):
            print("exit nan in pred loss!!!")
            print("sup_pred\n", sup_pred)
            sys.exit(0)
        
        losses.append(loss.item())
        loss.backward()
        
        sup_preds.append(sup_pred.detach().cpu().numpy())
        sup_labels.append(labels.detach().cpu().numpy())
        
        del out, loss, sup_pred, labels

        B, L = observed_mask.size(0), observed_mask.size(1)
        
        optimizer.step()

        del observed_data, observed_mask, observed_tp
        gc.collect()
        torch.cuda.empty_cache()


    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        acc, auroc, auprc, _, _, _ = evaluate_mc(sup_labels, sup_preds, opt.n_classes)

    return acc, auroc, auprc, train_loss

def eval_epoch(model, validation_data, pred_loss_func, opt, classifier, save_res=False):
    """ Epoch operation in evaluation phase. """

    model.eval()

    valid_losses = []
    sup_preds = []
    sup_labels = []
    acc, auroc, auprc = 0,0,0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
            train_batch, labels, seq_lens = map(lambda x: x.to(opt.device), batch)
            max_len = int(seq_lens.max().item())

            observed_data, observed_mask, observed_tp = \
                train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2*opt.num_types],\
                        train_batch[:, :max_len, 2*opt.num_types:3*opt.num_types]
            del train_batch
            
            out = model(observed_tp, observed_data, observed_mask, opt) # [B,L,K,D]
            sup_pred = classifier(out)

            if sup_pred.dim() == 1:
                sup_pred = sup_pred.unsqueeze(0)

            valid_loss = torch.sum(pred_loss_func((sup_pred + eps), labels))
            # sup_pred = torch.softmax(sup_pred, dim=-1)

            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(labels.detach().cpu().numpy())

            if valid_loss != 0:
                valid_losses.append(valid_loss.item())

            del out, observed_data, observed_mask, observed_tp, valid_loss

            gc.collect()
            torch.cuda.empty_cache()

    valid_loss = np.average(valid_losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels, axis=0)
        sup_preds = np.concatenate(sup_preds, axis=0)
        sup_preds = np.nan_to_num(sup_preds)

        # save prediction results
        if save_res:
            np.save(opt.save_res + '_prediction.npy',sup_preds)

    acc, auroc, auprc, precision, recall, F1 = evaluate_mc(sup_labels, sup_preds, opt.n_classes)
    return acc, auroc, auprc, precision, recall, F1, valid_loss

def run_experiment(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):

    epoch = 0
    best_valid_metric = -np.inf
    scaler = torch.cuda.amp.GradScaler()
    if not opt.test_only:
        """ Start training. """
        for epoch_i in range(opt.epoch):
            
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_acc, train_auroc, train_auprc, train_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler)
            log_info(opt, 'Train', epoch, train_acc, start=start, auroc=train_auroc, auprc=train_auprc, loss=train_loss, save=True)

            if not opt.retrain:
                start = time.time()
                valid_acc, valid_auroc, valid_auprc, valid_precision, valid_recall, valid_F1, valid_loss = eval_epoch(model, validation_data, pred_loss_func, opt, classifier)
                log_info(opt, 'Valid', epoch, valid_acc, auroc=valid_auroc, auprc=valid_auprc, start=start, precision=valid_precision, recall=valid_recall, F1=valid_F1, loss=valid_loss, save=True)
                
                start = time.time()
                if(best_valid_metric < valid_auroc):
                    best_valid_metric = valid_auroc
                    test_acc, test_auroc, test_auprc, test_precision, test_recall, test_F1, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)
                    log_info(opt, 'Testing', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, \
                             precision=test_precision, recall=test_recall, F1=test_F1, save=True)

                    best_epoch = epoch
                    best_test_acc = test_acc
                    best_test_auroc = test_auroc
                    best_test_auprc = test_auprc
                    best_test_precision = test_precision
                    best_test_recall = test_recall
                    best_test_F1 = test_F1

                log_info(opt, '* Best Testing *', best_epoch, best_test_acc, start=start, auroc=best_test_auroc, auprc=best_test_auprc,\
                        precision=best_test_precision, recall=best_test_recall, F1=best_test_F1, save=True)

                if early_stopping is not None:
                    early_stopping(-valid_auroc, model, classifier, epoch=epoch)

                    if early_stopping.early_stop: #and not opt.pretrain:
                        print("Early stopping. Training Done.")
                        break
            else:
                start = time.time()
                test_acc, test_auroc, test_auprc, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

                log_info(opt, 'Testing', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, save=True)

            scheduler.step()

    if not opt.retrain and save_path is not None:
        print("Testing...")
        model, classifier, _, _ = load_checkpoints(save_path, model, classifier=classifier, dp_flag=opt.dp_flag)

        start = time.time()
        test_acc, auroc, auprc, test_precision, test_recall, test_F1, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

        if early_stopping is not None and early_stopping.best_epoch > 0:
            best_epoch = early_stopping.best_epoch
        else:
            best_epoch = epoch
            
        log_info(opt, 'Testing', best_epoch, test_acc, start=start, auroc=auroc, auprc=auprc, \
                  precision=test_precision, recall=test_recall, F1=test_F1, save=True)

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--state', type=str, default='def')
    parser.add_argument('--model', type=str, default='gpt', help='select from [gpt, gpt_patch, warpformer]')

    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--data_path', type=str, default='./data/')

    parser.add_argument('-n',  type=int, default=12000, help="Size of the dataset")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_types', type=int, default=23)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log', type=str, default='./logs/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--task', type=str, default='nan')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fillmiss', action='store_true')
    parser.add_argument('--dp_flag', action='store_true')
    parser.add_argument('--load_in_batch', action='store_true')

    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--n_classes',  type=int, default=2)

    ### plm4ists
    parser.add_argument('--d_model', type=int, default=768, help="d_model of the PLM, 768 for Bert&GPT")
    parser.add_argument('--n_te_plmlayer', type=int, default=6)
    parser.add_argument('--n_st_plmlayer', type=int, default=6)
    parser.add_argument('--te_model', type=str, default='gpt')
    parser.add_argument('--st_model', type=str, default='bert')
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--semi_freeze', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--sample-tp', type=float, default=1.0)
    parser.add_argument('--collate', type=str, default='indseq')
    parser.add_argument('--zero_shot_age', action='store_true')
    parser.add_argument('--zero_shot_ICU', action='store_true')

    # dataset
    parser.add_argument('--split', type=str, default='1')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    seed = opt.seed

    opt.device = torch.device('cuda')
    # opt.device = torch.device('cpu')

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    print(input_command)

    setup_seed(seed)

    """ prepare dataloader """
    if opt.task == 'PAM':
        trainloader, validloader, testloader, opt.num_types, max_len = get_PAM_data(opt, opt.device)
        print("max_len:", max_len)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 8
        opt.input_dim = opt.num_types
    
    elif opt.task == 'P12':
        if opt.zero_shot_age or opt.zero_shot_ICU:
            trainloader, validloader, testloader, opt.num_types, max_len = get_P12_data_zeroshot(opt, opt.device)
        else:
            trainloader, validloader, testloader, opt.num_types, max_len = get_P12_data(opt, opt.device)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 2
        opt.input_dim = opt.num_types
    
    elif opt.task == 'P19':
        trainloader, validloader, testloader, opt.num_types, max_len = get_P19_data(opt, opt.device)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 2
        opt.input_dim = opt.num_types

    opt.log = opt.root_path + opt.log

    if opt.save_path is not None:
        opt.save_path = opt.root_path + opt.save_path
    
    if opt.load_path is not None:
        opt.load_path = opt.root_path + opt.load_path

    """ prepare model """
    if(opt.model == 'istsplm'):
        model = ists_plm(opt)
        
    elif(opt.model == 'istsplm_vector'):
        model = istsplm_vector(opt)

    elif(opt.model == 'istsplm_set'):
        model = istsplm_set(opt)


    print("! The backbone model is:", opt.model)

    para_list = list(model.parameters())

    if(opt.model == 'istsplm'):
        mort_classifier = Classifier(opt.d_model * opt.input_dim, opt.n_classes)
    else:
        mort_classifier = Classifier(opt.d_model, opt.n_classes)
    
    para_list += list(mort_classifier.parameters())
    
    # load model
    if opt.load_path is not None:
        print("Loading checkpoints...")
        model, mort_classifier, _, _ = load_checkpoints(opt.load_path, model, classifier=mort_classifier, dp_flag=False)

    
    model = model.to(opt.device)
    
    for mod in [model, mort_classifier]:
        if mod is not None:
            mod = mod.to(opt.device)
    
    if opt.dp_flag:
        model = nn.DataParallel(model)

    if opt.debug:
        opt.state='debug'
        exp_desc = f"{opt.task}_{opt.model}_{opt.state}"
    else:
        exp_desc = f"{opt.task}_{opt.model}_{opt.state}_plmlayer{opt.n_te_plmlayer}_{opt.n_st_plmlayer}_lr{opt.lr}"
    
    opt.log = f"{opt.log}{exp_desc}.log"
    print("! Log path:", opt.log)

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)
        opt.save_res = opt.save_path + exp_desc
        save_path = opt.save_path + exp_desc + '.h5'
        
    else:
        save_path = None
    
    """ optimizer and scheduler """

    params = (para_list)
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function """
    pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(opt.device)


    # setup the log file
    with open(opt.log, 'a') as f:
        f.write('[Info] parameters: {}\n'.format(opt))

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    print('[Info] parameters: {}'.format(opt))

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, save_path=save_path, dp_flag=opt.dp_flag)

    """ train the model """
    run_experiment(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, early_stopping, mort_classifier, save_path=save_path)

if __name__ == '__main__':
    main()
