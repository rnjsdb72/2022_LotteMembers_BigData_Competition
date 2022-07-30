import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import torch
import time
import os
from dataset import *
from utils import *

def train(model, optimizer, criterion, sampler, dataset, f, num_batch, epoch_start_idx, log_interval, args):
    T = 0.0
    t0 = time.time()
    best_hr = 0

    # 학습
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition

        pbar = tqdm(range(num_batch), total=num_batch)
        for step in pbar:
            u, seq, time_seq, time_matrix, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            if args.model.name == "TiSASRec":
                time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
            elif args.model.name == "TiSASRecwithAux":
                time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
                buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = np.array(buy_am), np.array(clac_hlv_nm), np.array(clac_mcls_nm), np.array(pd_nm), np.array(chnl_dv), np.array(de_dt_month), np.array(ma_fem_dv), np.array(ages), np.array(zon_hlv)

            if args.model.name == "TiSASRec":
                pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
            elif args.model.name == "SASRec":
                pos_logits, neg_logits = model(u, seq, pos, neg)
            elif args.model.name == "TiSASRecwithAux":
                pos_logits, neg_logits = model(u, seq, time_matrix, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv, pos, neg)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0

            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            if args.model.name != "SASRec":
                for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)

            loss.backward()
            optimizer.step()

            description = f'Epoch [{epoch}/{args.num_epochs}], Step [{step+1}/{num_batch}]: '
            description += f'running Loss: {round(loss.item(), 4)}'
            pbar.set_description(description)

        if epoch % log_interval == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            if args.validation == True:
                t_valid = evaluate_valid(model, dataset, args)
                print('Epoch:%d, Time: %f(s), Validation (NDCG@%d: %.4f, HR@%d: %.4f), Test (NDCG@%d: %.4f, HR@%d: %.4f)'
                    % (epoch, T, args.topk, t_valid[0], args.topk, t_valid[1], args.topk, t_test[0], args.topk, t_test[1]))
            else:
                print('Epoch:%d, Time: %f(s), Test (NDCG@%d: %.4f, HR@%d: %.4f)'
                        % (epoch, T, args.topk, t_test[0], args.topk, t_test[1]))
            
            if not os.path.isdir("../models/" + args.dataset + '_' + args.train_dir + '/results'):
                os.makedirs("../models/" + args.dataset + '_' + args.train_dir + '/results')

            if best_hr < t_test[1]:
                folder = "../models/" + args.dataset + '_' + args.train_dir + '/results'
                print(f"Best performance at epoch: {epoch}")
                print(f"Save model in {folder}")
                best_hr = t_test[1]

                if args.model.name == "TiSASRec":
                    fname = 'TiSASRec.total_epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.epoch={}.validation={}.pth'
                    fname = fname.format(args.num_epochs, args.optimizer.args.lr, args.model.args.num_blocks, 
                                    args.model.args.num_heads, args.model.args.hidden_units, args.model.args.maxlen, epoch, str(args.validation))
                elif args.model.name == "SASRec":
                    fname = 'SASRec.total_epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.epoch={}.validation={}.pth'
                    fname = fname.format(args.num_epochs, args.optimizer.args.lr, args.model.args.num_blocks, 
                                    args.model.args.num_heads, args.model.args.hidden_units, args.model.args.maxlen, epoch, str(args.validation))
                elif args.model.name == "TiSASRecwithAux":
                    fname = 'TiSASRecwithAux.total_epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.seq_attr_hidden_units={}.user_attr_emb_size={}.num_layers_user_aux={}.inner_size={}.fusion_type_item={}.fusion_type_final={}.epoch={}.validation={}.pth'
                    fname = fname.format(args.num_epochs, args.optimizer.args.lr, args.model.args.num_blocks, 
                                    args.model.args.num_heads, args.model.args.hidden_units, args.model.args.maxlen,
                                    args.model.args.seq_attr_hidden_units, args.model.args.user_attr_emb_size, args.model.args.num_layers_user_aux,
                                    args.model.args.inner_size, args.model.args.fusion_type_item, args.model.args.fusion_type_final, epoch, str(args.validation))
                torch.save(model.state_dict(), os.path.join(folder, fname))

                if len(os.listdir(folder)) > 3:
                    remove_old_files(folder, thres=3)

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

def evaluate(model, dataset, args):
    if args.validation == False:
        [train, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    elif args.validation == True:
        [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    
    users = range(1, usernum + 1)
    pbar = tqdm(users, total=len(users))
    for u in pbar:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name != "SASRec":
            time_seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name == "TiSASRecwithAux":
            buy_am = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_hlv_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_mcls_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            pd_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            chnl_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            de_dt_month = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ma_fem_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ages = np.zeros([args.model.args.maxlen], dtype=np.int32)
            zon_hlv = np.zeros([args.model.args.maxlen], dtype=np.int32)
        idx = args.model.args.maxlen - 1
        
        seq[idx] = test[u][0][0]
        if args.model.name != "SASRec":
            time_seq[idx] = test[u][0][1]
        if args.model.name == "TiSASRecwithAux":
            buy_am[idx] = test[u][0][2]
            clac_hlv_nm[idx] = test[u][0][3]
            clac_mcls_nm[idx] = test[u][0][4]
            pd_nm[idx] = test[u][0][5]
            chnl_dv[idx] = test[u][0][6]
            de_dt_month[idx] = test[u][0][7]
            ma_fem_dv[idx] = test[u][0][8]
            ages[idx] = test[u][0][9]
            zon_hlv[idx] = test[u][0][10]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            if args.model.name != "SASRec":
                time_seq[idx] = i[1]
            if args.model.name == "TiSASRecwithAux":
                buy_am[idx] = i[2]
                clac_hlv_nm[idx] = i[3]
                clac_mcls_nm[idx] = i[4]
                pd_nm[idx] = i[5]
                chnl_dv[idx] = i[6]
                de_dt_month[idx] = i[7]
                ma_fem_dv[idx] = i[8]
                ages[idx] = i[9]
                zon_hlv[idx] = i[10]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0],train[u]))
        if args.validation == True:
            rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
    
        if args.model.name != "SASRec":
            time_matrix = computeRePos(time_seq, args.model.args.time_span)

        if args.model.name == "SASRec":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        elif args.model.name == "TiSASRec":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        elif args.model.name == "TiSASRecwithAux":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [buy_am], [clac_hlv_nm], [clac_mcls_nm], [pd_nm], [chnl_dv], [de_dt_month], [ma_fem_dv], [ages], [zon_hlv], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, args):
    if args.validation == False:
        [train, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    elif args.validation == True:
        [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    users = range(1, usernum + 1)
    pbar = tqdm(users, total=len(users))
    for u in pbar:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name != "SASRec":
            time_seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name == "TiSASRecwithAux":
            buy_am = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_hlv_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_mcls_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            pd_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            chnl_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            de_dt_month = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ma_fem_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ages = np.zeros([args.model.args.maxlen], dtype=np.int32)
            zon_hlv = np.zeros([args.model.args.maxlen], dtype=np.int32)
        idx = args.model.args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i[0]
            if args.model.name != "SASRec":
                time_seq[idx] = i[1]
            if args.model.name == "TiSASRecwithAux":
                buy_am[idx] = i[2]
                clac_hlv_nm[idx] = i[3]
                clac_mcls_nm[idx] = i[4]
                pd_nm[idx] = i[5]
                chnl_dv[idx] = i[6]
                de_dt_month[idx] = i[7]
                ma_fem_dv[idx] = i[8]
                ages[idx] = i[9]
                zon_hlv[idx] = i[10]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        if args.model.name != "SASRec":
            time_matrix = computeRePos(time_seq, args.model.args.time_span)

        if args.model.name == "SASRec":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        elif args.model.name == "TiSASRec":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        elif args.model.name == "TiSASRecwithAux":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [buy_am], [clac_hlv_nm], [clac_mcls_nm], [pd_nm], [chnl_dv], [de_dt_month], [ma_fem_dv], [ages], [zon_hlv], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user