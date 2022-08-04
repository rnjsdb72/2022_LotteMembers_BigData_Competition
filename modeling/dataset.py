import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import joblib

def random_neq(l, r, s):
    """
    negative sampling code
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    """
    Caculate Relative Time Interval Matrix
    """
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(round((time_seq[i]-time_seq[j])/(60*60)))
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def computeRePos_c(time_seq, time_span, date=None):
    """
    Caculate Relative Time Interval with Current Date Matrix
    """
    size = time_seq.shape[0]
    if date != None:
        time_matrix = np.zeros([size], dtype=np.int32)
        for i in range(size):
            span = abs(round((date-time_seq[i])/(60*60)))
            if span > time_span:
                time_matrix[i] = time_span
            else:
                time_matrix[i] = span
        time_matrix = np.tile(time_matrix, (1,size))
    else:
        time_matrix = np.zeros([size, size], dtype=np.int32)
        for i in range(size-1):
            for j in range(size):
                span = abs(round(time_seq[i+1] - time_seq[j]))
                if span > time_span:
                    time_matrix[i] = time_span
                else:
                    time_matrix[i] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    """
    Caculate Relation Time Interval Embedding
    """
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def Relation_c(user_train, usernum, maxlen, time_span):
    """
    Caculate Relation Time Interval Embedding
    """
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos_c(time_seq, time_span)
    return data_train

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, relation_matrix_c, result_queue, SEED):
    def sample(user):
        """
        Sampling dataset
        """
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        buy_am = np.zeros([maxlen], dtype=np.float32)
        clac_hlv_nm = np.zeros([maxlen], dtype=np.int32)
        clac_mcls_nm = np.zeros([maxlen], dtype=np.int32)
        cop_c = np.zeros([maxlen], dtype=np.int32)
        chnl_dv = np.zeros([maxlen], dtype=np.int32)
        de_dt_month = np.zeros([maxlen], dtype=np.int32)
        ma_fem_dv = np.zeros([maxlen], dtype=np.int32)
        ages = np.zeros([maxlen], dtype=np.int32)
        zon_hlv = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]
    
        idx = maxlen - 1
        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            buy_am[idx] = i[2]
            clac_hlv_nm[idx] = i[3]
            clac_mcls_nm[idx] = i[4]
            cop_c[idx] = i[5]
            chnl_dv[idx] = i[6]
            de_dt_month[idx] = i[7]
            ma_fem_dv[idx] = i[8]
            ages[idx] = i[9]
            zon_hlv[idx] = i[10]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        time_matrix_c = relation_matrix_c[user]
        return (user, seq, time_seq, time_matrix, time_matrix_c, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, relation_matrix_c, batch_size=64, maxlen=10,n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      relation_matrix_c,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def timeSlice(time_set):
    """
    Scaling TimeStamps
    """
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    """
    Make user_set
    """
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], *x[2:]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1), *x[2:]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition_no_valid(fname):
    """
    Split Dataset
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('../data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    f = open('../data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        buy_am = float(buy_am)
        clac_hlv_nm = int(clac_hlv_nm)
        clac_mcls_nm = int(clac_mcls_nm)
        cop_c = int(cop_c)
        chnl_dv = int(chnl_dv)
        de_dt_month = int(de_dt_month)
        ma_fem_dv = int(ma_fem_dv)
        ages = int(ages)
        zon_hlv = int(zon_hlv)
        if user_count[u]<5 or item_count[i]<5: # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_test[user] = []
        else:
            user_train[user] = User[user][:-1]
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_test, usernum, itemnum, timenum]

def data_partition_with_valid(fname):
    """
    Split Dataset
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('../data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    f = open('../data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        buy_am = float(buy_am)
        clac_hlv_nm = int(clac_hlv_nm)
        clac_mcls_nm = int(clac_mcls_nm)
        cop_c = int(cop_c)
        chnl_dv = int(chnl_dv)
        de_dt_month = int(de_dt_month)
        ma_fem_dv = int(ma_fem_dv)
        ages = int(ages)
        zon_hlv = int(zon_hlv)
        if user_count[u]<5 or item_count[i]<5: # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-1]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]

def evaluate_dataloader_test(dataset, args):
    if args.validation == False:
        [train, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    elif args.validation == True:
        [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    print("Operate DataLoader,,,")
    results = {}
    users = range(1, usernum + 1)
    pbar = tqdm(users, total=len(users))
    for u in pbar:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
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
        
        if args.validation == True:
            seq[idx] = valid[u][0][0]
            time_seq[idx] = valid[u][0][1]
            buy_am[idx] = valid[u][0][2]
            clac_hlv_nm[idx] = valid[u][0][3]
            clac_mcls_nm[idx] = valid[u][0][4]
            cop_c[idx] = valid[u][0][5]
            chnl_dv[idx] = valid[u][0][6]
            de_dt_month[idx] = valid[u][0][7]
            ma_fem_dv[idx] = valid[u][0][8]
            ages[idx] = valid[u][0][9]
            zon_hlv[idx] = valid[u][0][10]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
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
        
        time_matrix = computeRePos(time_seq, args.model.args.time_span)
        time_matrix_c = computeRePos_c(time_seq, args.model.args.time_span)

        results[u] = [np.array(l) for l in [[u], [seq], [time_matrix], [time_matrix_c], [buy_am], [clac_hlv_nm], [clac_mcls_nm], [cop_c], [chnl_dv], [de_dt_month], [ma_fem_dv], [ages], [zon_hlv], item_idx]]

    joblib.dump(results, '../data/eval_dataloader_test_%s_%d_%d_%s.pickle'%(args.dataset, args.model.args.maxlen, args.model.args.time_span, str(args.validation)))

    return results

def evaluate_dataloader_valid(dataset, args):
    if args.validation == False:
        [train, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    elif args.validation == True:
        [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    print("Operate DataLoader,,,")
    results = {}
    users = range(1, usernum + 1)
    pbar = tqdm(users, total=len(users))
    for u in pbar:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.model.args.maxlen], dtype=np.int32)
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

        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
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
        
        time_matrix = computeRePos(time_seq, args.model.args.time_span)
        time_matrix_c = computeRePos_c(time_seq, args.model.args.time_span)

        results[u] = [np.array(l) for l in [[u], [seq], [time_matrix], [time_matrix_c], [buy_am], [clac_hlv_nm], [clac_mcls_nm], [cop_c], [chnl_dv], [de_dt_month], [ma_fem_dv], [ages], [zon_hlv], item_idx]]

    joblib.dump(results, '../data/eval_dataloader_valid_%s_%d_%d_%s.pickle'%(args.dataset, args.model.args.maxlen, args.model.args.time_span, str(args.validation)))

    return results

def inference_dataset(fname):
    """
    Split Dataset
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    dataset = {}
    
    print('Preparing data...')
    f = open('../data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    f = open('../data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        except:
            u, i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        buy_am = float(buy_am)
        clac_hlv_nm = int(clac_hlv_nm)
        clac_mcls_nm = int(clac_mcls_nm)
        cop_c = int(cop_c)
        chnl_dv = int(chnl_dv)
        de_dt_month = int(de_dt_month)
        ma_fem_dv = int(ma_fem_dv)
        ages = int(ages)
        zon_hlv = int(zon_hlv)
        if user_count[u]<5 or item_count[i]<5: # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp, buy_am, clac_hlv_nm, clac_mcls_nm, cop_c, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        dataset[user] = User[user]
    
    print('Preparing done...')
    return [dataset, usernum, itemnum, timenum]