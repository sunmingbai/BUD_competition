#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# In[2]:


# Data preparation


# In[3]:


historicalsale=pd.read_csv("E:\工作\百威\(Replace)Historical Sales Volume 2016.1-2019.11.csv")
masterdata=pd.read_csv("E:\工作\百威\(Replace)Product Master Data.csv")
commercialplanning=pd.read_csv("E:\工作\百威\(Replace)Commercial Planning.csv")


# In[110]:


saleseries=pd.read_csv("C:/Users/mings/BUD/rearranged_sale.csv")
calendar=pd.read_csv("E:\工作\百威\calendar.csv")


# In[5]:


historicalsale['Date']=historicalsale['Date'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d'))


# In[6]:


historicalsale.set_index('Date',inplace=True)


# In[907]:


historicalsale['Day']=historicalsale.index.map(lambda x: x.day)


# In[908]:


historicalsale


# In[952]:


# create new historical sale which will contain all the  zero
from datetime import datetime
saleseries['Unnamed: 0']=saleseries['Unnamed: 0'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
saleseries


# In[953]:


def seriestrans(saleseries):
    niters=saleseries.shape[1]
    clen=saleseries.shape[0]
    counts=0
    for i in range(1,niters):
        region=re.match('(.*)_(.*)',saleseries.columns[i]).group(1)
        sku=re.match('(.*)_(.*)',saleseries.columns[i]).group(2)
        for j in range(clen):
            hissaleall.loc[counts,'Year']=saleseries.iloc[j,0].year
            hissaleall.loc[counts,'Month']=saleseries.iloc[j,0].month
            hissaleall.loc[counts,'Day']=saleseries.iloc[j,0].year
            hissaleall.loc[counts,'Date']=saleseries.iloc[j,0]
            hissaleall.loc[counts,'SalesRegion']=region
            hissaleall.loc[counts,'SKU Code']=sku
            hissaleall.loc[counts,'VolumeHL']=saleseries.iloc[j,i]
            counts+=1


# In[954]:


hissaleall=pd.DataFrame(columns=['Date','Year','Month','Day','SalesRegion','SKU Code','VolumeHL'],index=range(130000))


# In[955]:


seriestrans(saleseries)


# In[956]:


hissaleall


# In[962]:


hissaleall.dropna(how='all',inplace=True)


# In[961]:


hissaleall.set_index('Date',inplace=True)


# In[964]:


hissaleall.fillna(0,inplace=True)


# In[970]:


hissaleall['SKU Code']=hissaleall['SKU Code'].astype(int)


# In[978]:


hissaleall['SalesRegion']=hissaleall['SalesRegion'].astype(str)


# In[979]:


hissaleall.info()


# In[9]:


# load temperature
Htem=pd.read_csv("E:\工作\百威\Data\HWeather_t.csv")
Jtem=pd.read_csv("E:\工作\百威\Data\JWeather_t.csv")
#Htem.info()
Htem


# In[11]:


import re


# In[12]:


Htem['Temp_max']=Htem['Temp_max'].map(lambda x: re.match('(.*)℃',x)[1])
Htem['Temp_max']=Htem['Temp_max'].astype(int)
Htem['Temp_min']=Htem['Temp_min'].map(lambda x: re.match('(.*)℃',x)[1])
Htem['Temp_min']=Htem['Temp_min'].astype(int)


# In[13]:


Htem['Date']=Htem['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[14]:


Htem.info()


# In[15]:


Jtem['Temp_min']=Jtem['Temp_min'].map(lambda x: re.match('(.*)℃',x)[1])
Jtem['Temp_max']=Jtem['Temp_max'].map(lambda x: re.match('(.*)℃',x)[1])
Jtem['Temp_max']=Jtem['Temp_max'].astype(int)
Jtem['Temp_min']=Jtem['Temp_min'].astype(int)
Jtem['Date']=Jtem['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[16]:


Htem.info()


# In[19]:


Htem['SalesRegion']='Heilongjiang'
Jtem['SalesRegion']='Jilin'


# In[49]:


Twotemps=Htem.append(Jtem)
Twotemps


# In[111]:


calendar.info()


# In[112]:


calendar['date']=calendar['date'].astype(str)
calendar['Date']=calendar['date'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d'))


# In[692]:


def labelevent(leadtimelist,endtimelist,markerlist,temdf):
    for i in range(len(leadtimelist)):
        tarevent=markerlist[i]
        print(tarevent)
        tarlead=leadtimelist[i]
        tarend=endtimelist[i]
        #print(1)
        if tarevent in ['National','Cultural']:
            #print(3)
            for m in range(len(temdf)):
                #print(temdf.loc[m,'event1'])
                if temdf.loc[m,'event_type_1']==tarevent:
                    mb=np.max([0,m-tarlead])
                    mf=np.max([0,m-tarend])
                    temdf.loc[mb:mf,tarevent]=1
                    #print(2)
        else:
            for m in range(len(temdf)):
                if temdf.loc[m,'event_type_2']==tarevent:
                    mb=np.max([0,m-tarlead])
                    mf=np.max([0,m-tarend])
                    temdf.loc[mb:mf,tarevent]=1
    return temdf[['National','Cultural','Sporting','Date']]


# In[114]:


calendar['event_type_2'].value_counts()


# In[115]:


calendar[calendar['event_type_2']=='COVID19']['event_type_2']=np.nan


# In[116]:


calendar['event_type_2'].value_counts()


# In[117]:


calendar['event_type_1'].value_counts()


# In[118]:


calendardroped=calendar.drop(columns=['event_name_1','holiday','work','event_name_1','weekday','wday','month','year','event_name_2'])


# In[119]:


calendardroped['National']=0
calendardroped['Cultural']=0
calendardroped['Sporting']=0


# In[120]:


calendardroped.info()


# In[547]:


HECPI=pd.read_csv("E:\工作\百威\Data\HCPI_E_t.csv")
JECPI=pd.read_csv("E:\工作\百威\Data\JCPI_E_t.csv")


# In[548]:


# a function to formalize  CPI data's format
def transcpi(tarset,region):
    try:
        tarset['Date']=tarset['Date'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d'))
    except:
        print('Already Datetime')
    tarset['Year']=tarset['Date'].map(lambda x: x.year)
    tarset['Month']=tarset['Date'].map(lambda x: x.month)
    tarset['SalesRegion']=region
    tarset.drop(columns=['Date'],inplace=True)
    return tarset


# In[549]:


HECPIT=transcpi(HECPI,'Heilongjiang')
JECPIT=transcpi(JECPI,'Jilin')
CPIT=HECPIT.append(JECPIT)


# In[556]:


CPITcols=[]
for x in CPIT.columns:
    if x not in ['Year','Month', 'SalesRegion']:
        CPITcols.append(x)
CPIT[CPITcols]=CPIT[CPITcols]-100
CPIT


# In[693]:


# a function for exponential smoothing and shift
def doews(tarset, alpha, periods):
    tarset=tarset.copy()
    for x in tarset.columns:
        if x not in ['Year','Month', 'SalesRegion','Date']:
            tarset[x+'_es_'+str(periods)]=tarset[x].ewm(alpha=alpha, adjust=False).mean()
            tarset[x+'_es_'+str(periods)]=tarset[x+'_es_'+str(periods)].shift(periods=periods)
            tarset=tarset.drop(columns=x)
    return tarset


# In[694]:


CPI_es_1=doews(CPIT, 0.7,1)
CPI_es_1


# In[1008]:


#  a function for finding coresponding historical data

# 后面可能需要考虑动态历史数据寻找
def findhistory(tarset):
    tarcol=[]
    for x in tarset.columns:
        if x not in ['Year','Month', 'SalesRegion','Date']:
            tarcol.append(x)
#     print(tarcol)
    tarcolfull=tarcol.copy().extend(['Month', 'SalesRegion'])
#     print(tarcol)
    tarset=tarset.copy()
    tarhist=pd.DataFrame(columns=tarcolfull,index=range(24))
    if 'Month' in tarset.columns:
        targrouped=tarset.groupby(['Month','SalesRegion'])
    elif 'Date' in tarset.columns:
        tarset['Month']=tarset['Date'].apply(lambda x: x.month)
        targrouped=tarset.groupby(['Month','SalesRegion'])
    else:
        print('Dont have proper timestamp')
    count=0
    for m in range(1,13):
        for n in ['Heilongjiang','Jilin']:
            tseries=targrouped.get_group((m,n)).mean()
            tarhist.loc[count,'Month']=m
            tarhist.loc[count,'SalesRegion']=n
            for x in tarcol:
#                 print(x)
                tarhist.loc[count,x]=tseries[x]
            count=count+1
#     mapdict={1:2,
#              2:3,
#              3:4,
#              4:5,
#              5:6,
#              6:7,
#              7:8,
#              8:9,
#              9:10,
#             10:11,
#             11:12,
#             12:1}
    tarhist['Month']=tarhist['Month'].apply(lambda x: x-1 if x>1 else x+11)
    tarhist.columns=tarhist.columns.map(lambda x: x+'_hist' if x in tarcol else x)
    return tarhist


# In[1009]:


CPIT_hisdata=findhistory(CPIT)


# In[1010]:


CPIT_hisdata


# In[1011]:


Temp_hisdata=findhistory(Twotemps)


# In[1012]:


Temp_hisdata


# In[773]:


sales=historicalsale.copy()
sales=sales.drop(columns=['Month','Year','YM'])
sales


# In[788]:


import datetime
def datarolling(tarset,rperiods,tarcol):
    tarset=tarset.copy()
    for i in range(len(rperiods)):
        tarset[tarcol+'_'+str(rperiods[i])]=tarset[tarcol].rolling(rperiods[i]).mean()
    tarset=tarset.drop(columns='VolumeHL')
    return tarset


# In[789]:


saleroll=datarolling(sales,[2,5,8,10,12,14,16,20,30],'VolumeHL')
saleroll.index=saleroll.index.map(lambda x: x+datetime.timedelta(days=1))


# In[790]:


saleroll


# In[909]:


def createdataset(totalset, historicalsale, masterdata, planning, Twotemps ,Temp_hisdata,CPIT,CPIT_hisdata, CPI_es_1,saleroll):
    totalset.index=historicalsale.index
    totalset[['Year','Month','Day','SalesRegion','SKU Code','VolumeHL']]=historicalsale[['Year','Month','Day','SalesRegion','SKU Code','VolumeHL']]
    totalset['Weekday']=totalset.index.map(lambda x:x.weekday())
    # add temperaature and other time series related features
    # temperature corresponding to date and region
    totalset=pd.merge(totalset,Twotemps,how='left',on=['Date','SalesRegion'])
    # set  event lagging use this function
    callabeled=labelevent([20,20,20],[15,15,15],['National','Cultural','Sporting'],calendardroped)
    totalset=pd.merge(totalset,callabeled,how='left',on=['Date'])
    totalset=pd.merge(totalset,CPIT,how='left',on=['Year','Month','SalesRegion'])
    totalset=pd.merge(totalset,Temp_hisdata,how='left',on=['Month','SalesRegion'])
    totalset=pd.merge(totalset,CPIT_hisdata,how='left',on=['Month','SalesRegion'])
    totalset=pd.merge(totalset,CPI_es_1,how='left',on=['Year','Month','SalesRegion'])
    totalset=pd.merge(totalset,saleroll,how='left',on=['Date','SKU Code','SalesRegion'])
    return totalset


# In[1013]:


# totalset=pd.DataFrame(index=range(len(historicalsale)))
totalset=pd.DataFrame(index=range(len(hissaleall)))

#参数说明
# totalset 目标数据集，用来呈现合并后的结构
# historicalsale 原始 历史销售数据集
# masterdata 产品主数据
# commercialplanning 产品商业计划数据 如促销\折扣等情况
# Twotemps 两省原始温度数据
# Temp_hisdata 历史同期温度数据
# CPIT 两省原始居民消费指数
# CPIT_hisdata 历史同期居民消费指数数据\
# saleroll rolling之后的salesvolumne

# totalset=createdataset(totalset, historicalsale, masterdata, commercialplanning,Twotemps ,Temp_hisdata,CPIT,CPIT_hisdata,CPI_es_1,saleroll)
totalset=createdataset(totalset, hissaleall, masterdata, commercialplanning,Twotemps ,Temp_hisdata,CPIT,CPIT_hisdata,CPI_es_1,saleroll)


# In[1014]:


totalset


# In[1015]:


totalsetdroped=totalset.copy()
totalsetdroped=totalsetdroped.dropna(how='any')


# In[913]:


from xgboost import XGBRegressor
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[991]:


totalsetdroped.describe()


# In[1016]:


def finalpre(tarset):
    tarset.replace({'SalesRegion':{'Heilongjiang':1,'Jilin':2}},inplace=True)
    xcols=[]
    for col in tarset.columns:
        if col not in ['VolumeHL','Date']:
            xcols.append(col)
    return xcols


# In[1017]:


xcols=finalpre(totalsetdroped)


# In[1018]:


xcols


# In[808]:


totalsetdroped


# In[717]:


def wmape( y_true,y_pre):
    y_true=pd.Series(y_true)
    y_pre=pd.Series(y_pre)
    try :
        len( y_true)==len(y_pre)
    except:
        print('Length is not equal')
    aerror=[np.abs(y_pre.iloc[i]-y_true.iloc[i]) for i in range(len(y_pre))]
    return np.sum(aerror)/np.sum(y_true)


# In[718]:


from sklearn.metrics import make_scorer


# In[719]:


from sklearn.model_selection import train_test_split
import xgboost  as xgb


# In[994]:


totalseth=totalsetr[totalsetr['SalesRegion']==1]
totalseth1=totalseth[totalseth['SKU Code']==1]
totalseth1


# In[995]:


totalsethd=totalsetdroped[totalsetdroped['SalesRegion']==1]
totalsetdh1=totalsethd[totalsethd['SKU Code']==1]
totalsetdh1


# # 分隔，上述为训练用初始数据集准备

# In[875]:


def trainingset(finalset,cols):
    X=finalset[cols]
    y=finalset['VolumeHL']
    return X,y


# In[1030]:


param = { 'eta': 0.5, 'silent': 1, 'objective':'reg:linear'}
#eta shrinks the feature weights to make the boosting process more conservative.
# the larger gamma is, the more conservative the model is
param['gama']=0
param['min_child_weight']=1

#Increasing this value will make the model more complex and more likely to overfit
param['max_depth']=6

param['seed'] = 100
param['eval_metric'] = ['mae']
#param['scale_pos_weight']=5
# try to avoid overfitting
param['colsample_bytree']=1
param['subsample']=1
# the larger gamma is, the more conservative the model is
param['max_delta_step']=0
param['lambda']=1
param['alpha']=0

num_round=60
num_fold=5


# In[865]:


import xgboost as xgb


# In[1025]:


bstr = xgb.XGBRegressor(**param)


# In[893]:


def drawimportance(feature_impor,cols):
    fi=pd.DataFrame(feature_impor,index=cols,columns=['gain score'])
    fi_ordered=fi.sort_values(by='gain score', ascending=False)

    fig=plt.figure(figsize=(20,10))
    ax=fig.add_subplot(111)

    ax.bar(fi_ordered.index,fi_ordered['gain score'])
    ax.set_ylabel('Gain Score',fontsize=20)
    ax.set_xlabel('Features',fontsize=20)
    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels,rotation=90,fontsize=10)

    axt=ax.twinx()
    p_n = 1.0*fi_ordered.cumsum()/fi_ordered.sum()
    axt.plot(p_n.index,p_n['gain score'],'-or')
    axt.set_ylabel('Accumulated Percentage',fontsize=20)
    return fi_ordered


# In[1028]:


def model_xgb(bstr, totalsetdh1, xcols, param):
    X,y=trainingset(totalsetdroped,xcols)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,
                                                    #stratify=patients_outcome['In-hospital_death'],
                                                    random_state=1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    xtest = xgb.DMatrix(X_test)
    result = xgb.cv(param, dtrain, num_round, num_fold)
    print(result)
    bstr.fit(X_train,y_train)
    xgb_pre=bstr.predict(X_test)
    out=wmape(y_test,xgb_pre)
    print('wmape is ',out)
    feature_i=bstr.feature_importances_
    fo=drawimportance(feature_i,xcols)
    return fo


# In[1029]:


featureis=model_xgb(bstr, totalsetdroped, xcols, param)


# In[1031]:


colsfilter2=featureis.index[:10].to_list()
colsfilter2


# In[1032]:


featureis1=model_xgb(bstr, totalsetdroped, colsfilter2, param)


# In[ ]:





# In[ ]:




