import os
import torch
import pickle
import json
from collections import namedtuple
from importlib import import_module

from dataset import *
from train import *
from utils import *

def main():
    # argument 정보 파일 저장
    args_ = arg_parse()
    with open(args_.cfg, 'r') as f:
        args = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    if not os.path.isdir("../models/" + args.dataset + '_' + args.train_dir):
        os.makedirs("../models/" + args.dataset + '_' + args.train_dir)
    with open(os.path.join("../models/" + args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(args._asdict().items(), key=lambda x: x[0])]))
    f.close()

    # seed 고정
    fix_seed(args.seed)

    # 데이터 불러오기
    if args.validation == False:
        dataset = data_partition_no_valid(args.dataset)
        [user_train, user_test, usernum, itemnum, timenum] = dataset
    elif args.validation == True:
        dataset = data_partition_with_valid(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # 로그 파일 불러오기
    f = open(os.path.join("../models/" + args.dataset + '_' + args.train_dir, 'log.txt'), 'a')

    # relation_matrix가 있으면 불러오고 없으면 생성
    try:
        relation_matrix = pickle.load(open('../data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.model.args.maxlen, args.model.args.time_span),'rb'))
    except:
        relation_matrix = Relation(user_train, usernum, args.model.args.maxlen, args.model.args.time_span)
        pickle.dump(relation_matrix, open('../data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.model.args.maxlen, args.model.args.time_span),'wb'))

    # 모델과 샘플러 불러오기
    sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.model.args.maxlen, n_workers=3)
    model_module = getattr(import_module("model"), args.model.name)
    if args.model.name != 'SASRec':
        model = model_module(usernum, itemnum, itemnum, args).to(args.device)
    else:
        model = model_module(usernum, itemnum, args).to(args.device)

    # 모델 초기화
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model.train() # enable model training

    # 희망 할 시, 모델 파라미터 불러오기
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            epoch_start_idx = int(args.state_dict_path.split('=')[-2].split('.')[0])+1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    # 추론만 희망할 경우, 추론만 시행
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@%d: %.4f, HR@%d: %.4f)' % (args.topk, t_test[0], args.topk, t_test[1]))

    # criterion, optimizer 불러오기
    criterion_module = getattr(import_module("torch.nn"), args.criterion.name)
    criterion = criterion_module(**args.criterion.args._asdict())

    optimizer_module = getattr(import_module("torch.optim"), args.optimizer.name)
    optimizer = optimizer_module(model.parameters(), **args.optimizer.args._asdict())

    # 학습
    train(model, optimizer, criterion, sampler, dataset, f, num_batch, epoch_start_idx, args.log_interval, args)

    f.close()
    sampler.close()
    print("Done")

if __name__ == '__main__':
    main()