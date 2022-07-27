import pandas as pd
import time
import os
from sklearn.preprocessing import RobustScaler

def make_df(prod, serv, cust, prod_info):
    lst = ['cust', 'rct_no', 'cop_c', 'chnl_dv', 'de_dt', 'de_hr', 'buy_am']
    prod['buy_am'] = prod['buy_am'] / prod['buy_ct']
    trans = pd.concat([prod[lst+['pd_c']], serv[lst]])
    trans.pd_c = trans.pd_c.fillna(trans.cop_c)
    
    trans = pd.merge(trans, cust, on='cust', how='left')
    
    trans = pd.merge(trans, prod_info, on='pd_c', how='left')
    trans.pd_nm = trans.pd_nm.fillna(trans.cop_c)
    trans.clac_hlv_nm = trans.clac_hlv_nm.fillna(trans.cop_c)
    trans.clac_mcls_nm = trans.clac_mcls_nm.fillna(trans.cop_c)
    trans = trans.sort_values(by=['cust','de_dt' , 'de_hr']).reset_index(drop = True )
    trans['de_dt_month'] = trans['de_dt']//100%100
    trans['de_dt_hr'] = trans['de_dt'].astype('str') + trans['de_hr'].astype('str')
    trans['de_dt_hr'] = pd.to_datetime( trans['de_dt_hr'] , format = '%Y%m%d%H' )
    trans['timestamp'] = trans['de_dt_hr'].apply(lambda x : time.mktime(x.timetuple())).astype("int32")
    lst2 = ['cust','pd_c','timestamp','buy_am','clac_hlv_nm','clac_mcls_nm','pd_nm','chnl_dv','de_dt_month','ma_fem_dv','ages','zon_hlv']
    trans = trans[lst2]
    
    transformer = RobustScaler()
    trans['buy_am'] = pd.DataFrame(transformer.fit_transform(trans[['buy_am']]) , columns = ['buy_am'])
    
    for i in lst2[:2] + lst2[4:] : 
        id2 = dict(enumerate(sorted(trans[i].unique())))
        id2 = {j:i for i, j in id2.items()}
        trans[i] = trans[i].map(lambda x: id2[x]+1)
    
    trans = trans.sort_values(by=['cust', 'timestamp'] ,ascending=True)

    if not os.path.exists("./data"):
        os.makedirs("./data")
    trans.drop_duplicates(inplace = True)
    trans.to_csv("./data/lpay_rec_dataset.txt", header = False, index = False, sep = "\t")
    print('Save Complete!')
    return trans

if __name__ == "__main__":
    yes = input('데이터 경로를 직접 입력을 희망하시면 "yes"를 입력해주세요: \t')
    if yes == "yes":
        prod = pd.read_csv(input('상품 구매 정보 데이터 csv 경로를 입력하시오: \t'))
        serv = pd.read_csv(input('제휴사 이용 정보 csv 경로를 입력하시오: \t'))
        cust = pd.read_csv(input('고객 데모 정보 데이터 csv 경로를 입력하시오: \t'))
        prod_info = pd.read_csv(input('상품 분류 정보 데이터 csv 경로를 입력하시오: \t'))
    else:
        prod = pd.read_csv('data/LPOINT_BIG_COMP_02_PDDE.csv')
        serv = pd.read_csv('data/LPOINT_BIG_COMP_03_COP_U.csv')
        cust = pd.read_csv('data/LPOINT_BIG_COMP_01_DEMO.csv')
        prod_info = pd.read_csv('data/LPOINT_BIG_COMP_04_PD_CLAC.csv')

    df = make_df(prod, serv, cust, prod_info)