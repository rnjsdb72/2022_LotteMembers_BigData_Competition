import pandas as pd
import time
import os
import joblib
import argparse
from sklearn.preprocessing import RobustScaler

def make_df(prod, serv, cust, prod_info, scaler):
    lst = ['cust', 'rct_no', 'cop_c', 'chnl_dv', 'de_dt', 'de_hr', 'buy_am']

    prod['buy_am'] = prod['buy_am'] / prod['buy_ct']
    trans = pd.concat([prod[lst+['pd_c']], serv[lst]])
    trans.pd_c = trans.pd_c.fillna(trans.cop_c)

    trans = pd.merge(trans, cust, on='cust', how='left')    
    trans = pd.merge(trans, prod_info, on='pd_c', how='left')

    trans.pd_nm = trans.pd_nm.fillna(trans.cop_c)
    trans.clac_hlv_nm = trans.clac_hlv_nm.fillna(trans.cop_c)
    trans.clac_mcls_nm = trans.clac_mcls_nm.fillna(trans.cop_c)

    trans['de_dt_month'] = trans['de_dt']//100%100
    trans['de_dt_hr'] = trans['de_dt'].astype('str') + trans['de_hr'].astype('str')
    trans['de_dt_hr'] = pd.to_datetime( trans['de_dt_hr'] , format = '%Y%m%d%H' )
    trans['timestamp'] = trans['de_dt_hr'].apply(lambda x : time.mktime(x.timetuple())).astype("int32")

    trans = trans.sort_values(by=['cust','de_dt' , 'de_hr']).reset_index(drop = True )

    trans['pd_c'] = trans[['cop_c', 'pd_c']].apply(lambda x: '_'.join(x), axis=1)

    lst2 = ['cust','pd_c','timestamp','buy_am','clac_hlv_nm','clac_mcls_nm','cop_c','chnl_dv','de_dt_month','ma_fem_dv','ages','zon_hlv']
    trans = trans[lst2]

    if scaler != None:
        transformer = joblib.load(scaler)
        trans['buy_am'] = pd.DataFrame(transformer.transform(trans[['buy_am']]) , columns = ['buy_am'])
    else:
        transformer = RobustScaler()
        trans['buy_am'] = pd.DataFrame(transformer.fit_transform(trans[['buy_am']]) , columns = ['buy_am'])
        if not os.path.exists('./models'):
            os.mkdir('./models')
        joblib.dump(transformer, 'models/RobustScaler.pkl')

    ids = []
    for i in lst2[:2] + lst2[4:] : 
        id2 = dict(enumerate(sorted(trans[i].unique())))
        ids.append({i:id2})
        id2 = {j:i for i, j in id2.items()}
        trans[i] = trans[i].map(lambda x: id2[x]+1)
    
    trans = trans.sort_values(by=['cust', 'timestamp'] ,ascending=True)
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
    joblib.dump(ids, "./data/id2cat_cols.pkl")
    trans.drop_duplicates(inplace = True)
    trans.to_csv("./data/lpay_rec_dataset.txt", header = False, index = False, sep = "\t")
    print('Save Complete!')
    return trans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prod', type=str, default='data/LPOINT_BIG_COMP_02_PDDE.csv')
    parser.add_argument('--serv', type=str, default='data/LPOINT_BIG_COMP_03_COP_U.csv')
    parser.add_argument('--cust', type=str, default='data/LPOINT_BIG_COMP_01_DEMO.csv')
    parser.add_argument('--prod_info', type=str, default='data/LPOINT_BIG_COMP_04_PD_CLAC.csv')
    parser.add_argument('--scaler', type=str, default='')

    args = parser.parse_args()

    if args.scaler == '':
        scaler = None
    else:
        scaler = args.scaler
    prod = pd.read_csv(args.prod)
    serv = pd.read_csv(args.serv)
    cust = pd.read_csv(args.cust)
    prod_info = pd.read_csv(args.prod_info)

    df = make_df(prod, serv, cust, prod_info, scaler)