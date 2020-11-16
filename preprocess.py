# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:57:20 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

filename = "mdr_20060118_SPX.mat"
with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
data = np.stack(data,axis=0).T
df = pd.DataFrame(data)
colname = ['recordType','tradeDate','tradeTime','series_seq',
           'mktcode','clscode','expiryDate','putcallType','exercisePrice',
           'bidPrice','bidSize','askPrice','askSize','instrumentPrice']
df.columns = colname
df = df[(df['tradeTime']>=82900) & (df['tradeTime']<=150000) & (df['recordType']==0) 
          & (df['instrumentPrice']>0) & (df['expiryDate']>df['tradeDate']) 
          & (df['bidPrice']>0) & (df['askPrice']>0)]
df = df[np.invert((((df['bidPrice']+df['askPrice'])/2 > df['instrumentPrice']) & (df['putcallType']==0)) | 
        (((df['bidPrice']+df['askPrice'])/2 > df['exercisePrice']) & (df['putcallType']==1)))]
clscodes = np.array([21,23,24,25,26,30,34,35,38,39,40,42,43,44,48,51]) - 1
df = df[df['clscode'].isin(list(clscodes))]
df['price'] = (df['askPrice'] + df['bidPrice']) / 2
cut_id = 10000*np.floor(np.arange(34,61,1)/4)+1500*(np.arange(34,61,1)%4)
df['TimeInt'] = pd.cut(df['tradeTime'],cut_id)
start = np.array((df['tradeDate']-366).astype('int64').apply(datetime.fromordinal).dt.date)
end = np.array((df['expiryDate']-366).astype('int64').apply(datetime.fromordinal).dt.date)
busT2M = np.array([np.busday_count(start[i],end[i]) for i in range(start.shape[0])])
df['busT2M'] = busT2M

filename = "Div_19870130_20150601.mat"
file = loadmat(filename)
div = file['SPX_Div']
div = pd.DataFrame(div)
div.columns = ['date','div','refer']
div['div'] = np.log(1+div['div'])
df = df.merge(right=div,how='left',left_on='tradeDate',right_on='date')
df = df.drop(['date','refer'],axis=1)

filename = "Rates_19870102_20140829.mat"
file = loadmat(filename)
rates = file['rates']
rates = pd.DataFrame(rates)
rates.columns = ['date','busT2M','y','refer']
rates = rates[rates['y']<=0.4]

#np.unique(busT2M)
#pd.unique(rates[rates['date']==732695]['busT2M'])
#df.merge(right=rates,how='left',left_on=['tradeDate','busT2M'],right_on=['date','busT2M'])
df['rate'] = 0.0434

df['moneyness'] = np.log(df['exercisePrice']/(df['instrumentPrice']*np.exp((df['rate']-df['div'])*df['busT2M']/252)))

# -------------------------------------------------------------------------------------
# calculate implied volatility

iv = np.zeros((df.shape[0]))
for i in range(df.shape[0]):
    temp_row = df.iloc[i]
    price = temp_row['price']
    S = temp_row['instrumentPrice']
    K = temp_row['exercisePrice']
    T = temp_row['busT2M']/252
    r = temp_row['rate']
    q = temp_row['div']
    if (temp_row['putcallType']==1):
        flag = 'p'
    else:
        flag = 'c'
    
    try:
        iv[i] = implied_volatility(price,S,K,T,r,q,flag)
    except:
        iv[i] = -1
        print('Degenerate IV at: ' + str(i))

df['iv'] = iv
df = df[df['iv']!=-1]

df.groupby('TimeInt').size()
#TimeInt
#(83000.0, 84500.0]      11850
#(84500.0, 90000.0]      11649
#(90000.0, 91500.0]      11883
#(91500.0, 93000.0]      15163
#(93000.0, 94500.0]       9995
#(94500.0, 100000.0]      9029
#(100000.0, 101500.0]    10496
#(101500.0, 103000.0]    10520
#(103000.0, 104500.0]    11127
#(104500.0, 110000.0]    11643
#(110000.0, 111500.0]    11174
#(111500.0, 113000.0]    12617
#(113000.0, 114500.0]    10796
#(114500.0, 120000.0]     8646
#(120000.0, 121500.0]     8474
#(121500.0, 123000.0]    11687
#(123000.0, 124500.0]    10872
#(124500.0, 130000.0]     9032
#(130000.0, 131500.0]    13281
#(131500.0, 133000.0]     9933
#(133000.0, 134500.0]    12111
#(134500.0, 140000.0]    10778
#(140000.0, 141500.0]    12165
#(141500.0, 143000.0]    13656
#(143000.0, 144500.0]    10474
#(144500.0, 150000.0]    12128
#dtype: int64

df['totalvar'] = df['iv']**2*df['busT2M']/252
df['busT2M'] = df['busT2M']/252
df.to_csv('modified_df.csv',sep=',',header=True)