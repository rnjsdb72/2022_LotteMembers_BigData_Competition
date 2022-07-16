import pandas as pd

def make_df(prod, serv, cust, prod_info):
    lst = ['cust', 'rct_no', 'cop_c', 'chnl_dv', 'de_dt', 'de_hr', 'buy_am']
    
    trans = pd.concat([prod[lst+['pd_c']], serv[lst]])
    trans.pd_c = trans.pd_c.fillna(trans.cop_c)
    
    trans = pd.merge(trans, cust, on='cust', how='left')
    
    trans = pd.merge(trans, prod_info, on='pd_c', how='left')
    trans.pd_nm = trans.pd_nm.fillna(trans.cop_c)
    trans.clac_hlv_nm = trans.clac_hlv_nm.fillna(trans.cop_c)
    trans.clac_mcls_nm = trans.clac_mcls_nm.fillna(trans.cop_c)
    trans = trans.sort_values(by=['cust','de_dt' , 'de_hr']).reset_index(drop = True )
    trans['de_dt_hr'] = trans['de_dt'].astype('str') + trans['de_hr'].astype('str')
    trans['de_dt_hr'] = pd.to_datetime( trans['de_dt_hr'] , format = '%Y%m%d%H' )
    trans['timestamp'] = trans['de_dt_hr'].apply(lambda x : time.mktime(x.timetuple()))

    return trans

if __name__ == "__main__":
    prod = pd.read_csv(input('상품 구매 정보 데이터 csv 경로를 입력하시오: \t'))
    serv = pd.read_csv(input('제휴사 이용 정보 csv 경로를 입력하시오: \t'))
    cust = pd.read_csv(input('고객 데모 정보 데이터 csv 경로를 입력하시오: \t'))
    prod_info = pd.read_csv(input('상품 분류 정보 데이터 csv 경로를 입력하시오: \t'))

    df = make_df(prod, serv, cust, prod_info)