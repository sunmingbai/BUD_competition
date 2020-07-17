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
historicalsale


# In[7]:


masterdata


# In[8]:


commercialplanning


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


# In[113]:


def labelevent(leadtimelist,markerlist,temdf):
    for i in range(len(leadtimelist)):
        tarevent=markerlist[i]
        print(tarevent)
        tarlead=leadtimelist[i]
        tarend=5
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


# In[132]:


def createdataset(totalset, historicalsale, masterdata, planning, Twotemps, rolling, lagging):
    totalset.index=historicalsale.index
    totalset[['Year','Month','SalesRegion','SKU Code','VolumeHL']]=historicalsale[['Year','Month','SalesRegion','SKU Code','VolumeHL']]
    totalset['Weekday']=totalset.index.map(lambda x:x.weekday())
    # add temperaature and other time series related features
    # temperature corresponding to date and region
    totalset=pd.merge(totalset,Twotemps,how='left',on=['Date','SalesRegion'])
    # set  event lagging use this function
    callabeled=labelevent([20,20,20],['National','Cultural','Sporting'],calendardroped)
    totalset=pd.merge(totalset,callabeled,how='left',on=['Date'])
    return totalset


# In[133]:


totalset=pd.DataFrame(index=range(len(historicalsale)))
createdataset(totalset, historicalsale, masterdata, commercialplanning,Twotemps ,5, 6)


# In[128]:


def laggingfeatures(totalset,lagdict):
    


# In[129]:


laggingfeature(totalset)


# In[145]:


from xgboost import XGBRegressor
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[148]:


totalsetr=totalset.replace({'SalesRegion':{'Heilongjiang':1,'Jilin':2}})


# In[149]:


xcols=[]
for col in totalset.columns:
    if col not in ['VolumeHL','Date']:
        xcols.append(col)
xcols


# In[150]:


X=totalsetr[xcols]
y=totalsetr['VolumeHL']


# In[157]:


X.info()


# In[158]:


model=XGBRegressor()
cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[170]:


def wmape( y_true,y_pre):
    aerror=[np.abs(y_pre[i]-y_true[i]) for i in range(len(y_pre))]
    return np.sum(aerror)/np.sum(y_true)


# In[173]:


from sklearn.metrics import make_scorer


# In[174]:


scores = cross_val_score(model, X, y, scoring=make_scorer(wmape, greater_is_better=False), cv=cv, n_jobs=-1)
# summarize performance
print('Mean wMAPE: %.5f' % mean(scores))


# In[179]:


from sklearn.model_selection import train_test_split
import xgboost  as xgb


# In[195]:


totalsetr


# In[196]:


totalseth=totalsetr[totalsetr['SalesRegion']==1]
totalseth1=totalseth[totalseth['SKU Code']==1]
totalseth1


# In[201]:


X=totalseth1[xcols]
y=totalseth1['VolumeHL']
X


# In[357]:


def trainingset(finalset):
    X=finalset[xcols]
    y=finalset['VolumeHL']
    return X,y


# In[361]:


X,y=trainingset(totalsetr)


# In[362]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,
                                                    #stratify=patients_outcome['In-hospital_death'],
                                                    random_state=1)


# In[363]:


dtrain = xgb.DMatrix(X_train, label=y_train)
xtest = xgb.DMatrix(X_test)


# In[364]:


param = { 'eta': 0.3, 'silent': 1, 'objective':'reg:linear'}
#eta shrinks the feature weights to make the boosting process more conservative.
# the larger gamma is, the more conservative the model is
param['gama']=0
param['min_child_weight']=1

#Increasing this value will make the model more complex and more likely to overfit
param['max_depth']=4

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

result = xgb.cv(param, dtrain, num_round, num_fold)
# pre=bst.predict(xtest)

# print(np.sqrt(mean_squared_error(y_true=y_test,y_pred=pre)))
print(result)


# In[366]:


import xgboost as xgb


# In[367]:


bstr = xgb.XGBRegressor(**param)
bstr.fit(X_train,y_train)


# In[368]:


xgb_pre=bstr.predict(X_test)


# In[369]:


wmape(y_test,xgb_pre)


# In[370]:


y_test


# In[371]:


xgb_pre


# In[ ]:




