from datetime import date
from tqdm import tqdm
import json
from collections import namedtuple
from importlib import import_module
from datetime import datetime
import time

import pandas as pd
import numpy as np
import torch

from dataset import *
from utils import *

def inference(model, dataset, args):
    dataset, usernum, itemnum, timenum = dataset

    users = range(1, usernum + 1)
    pbar = tqdm(users, total=len(users))

    results = []
    for u in pbar:
        seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name != "SASRec":
            time_seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        if args.model.name == "TiSASRecwithAux":
            buy_am = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_hlv_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            clac_mcls_nm = np.zeros([args.model.args.maxlen], dtype=np.int32)
            cop_c = np.zeros([args.model.args.maxlen], dtype=np.int32)
            chnl_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            de_dt_month = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ma_fem_dv = np.zeros([args.model.args.maxlen], dtype=np.int32)
            ages = np.zeros([args.model.args.maxlen], dtype=np.int32)
            zon_hlv = np.zeros([args.model.args.maxlen], dtype=np.int32)
        idx = args.model.args.maxlen - 1

        for i in reversed(dataset[u]):
            seq[idx] = i[0]
            if args.model.name != "SASRec":
                time_seq[idx] = i[1]
            if args.model.name == "TiSASRecwithAux":
                buy_am[idx] = i[2]
                clac_hlv_nm[idx] = i[3]
                clac_mcls_nm[idx] = i[4]
                cop_c[idx] = i[5]
                chnl_dv[idx] = i[6]
                de_dt_month[idx] = i[7]
                ma_fem_dv[idx] = i[8]
                ages[idx] = i[9]
                zon_hlv[idx] = i[10]
            idx -= 1
            if idx == -1: break

        rec_date = time.mktime(datetime.strptime(args.date, '%Y-%m-%d %H:%M:%S').timetuple())

        time_matrix = computeRePos(time_seq, args.model.args.time_span)
        time_matrix_c = computeRePos_c(time_seq, args.model.args.time_span, rec_date)
        
        item_idx = list(range(itemnum+1))
        if (args.model.name == "SASRec" or args.model.name == "TiSASRec"):
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        elif args.model.name == "TiSASReconlyCTI":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix_c], item_idx]])
        elif args.model.name == "TiSASRecwithCTI":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_matrix_c], item_idx]])
        elif args.model.name == "TiSASRecwithAux":
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_matrix], [buy_am], [clac_hlv_nm], [clac_mcls_nm], [cop_c], [chnl_dv], [de_dt_month], [ma_fem_dv], [ages], [zon_hlv], item_idx]])    

        pred_item = predictions[0].argsort()[:args.topk]
        results.append((u, pred_item))

    return results

def main():
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        args = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    print("Load Dataset&Model")
    dataset = inference_dataset(args.dataset)
    dataset_, usernum, itemnum, timenum = dataset
    model_module = getattr(import_module("model"), args.model.name)
    if args.model.name != 'SASRec':
        model = model_module(usernum, itemnum, itemnum, args).to(args.device)
    else:
        model = model_module(usernum, itemnum, args).to(args.device)
    if args.state_dict_path is not None:
        #try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        #except:
        #    print('failed loading state_dicts, pls check file path: ', end="")
        #    exit()

    print("Start Inference!")
    res = inference(model, dataset, args)
    x = dict(map(lambda x: (x[0], x[1].tolist()), res))
    df = pd.DataFrame(res)
    df.T.rename(columns={x:'top'+str(x) for x in df.columns}).to_csv(f'../data/inference_res_{args.train_dir}')
    print("Save Complete!")

if __name__ == "__main__":
    main()