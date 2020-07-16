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


# In[57]:


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


# In[54]:


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
                if temdf.loc[m,'event1']==tarevent:
                    mb=np.max([0,m-tarlead])
                    mf=np.max([0,m-tarend])
                    temdf.loc[mb:mf,tarevent]=1
                    #print(2)
        else:
            for m in range(len(temdf)):
                if temdf.loc[m,'event2']==tarevent:
                    mb=np.max([0,m-tarlead])
                    mf=np.max([0,m-tarend])
                    temdf.loc[mb:mf,tarevent]=1
    return temdf


# In[60]:


calendar.drop


# In[50]:


def createdataset(totalset, historicalsale, masterdata, planning, Twotemps, rolling, lagging):
    totalset.index=historicalsale.index
    totalset[['Year','Month','SalesRegion','SKU Code']]=historicalsale[['Year','Month','SalesRegion','SKU Code']]
    totalset['Weekday']=totalset.index.map(lambda x:x.weekday())
    # add temperaature and other time series related features
    # temperature corresponding to date and region
    totalset=pd.merge(totalset,Twotemps,how='left',on=['Date','SalesRegion'])
    # so we can add more features just like this
    
    return totalset


# In[52]:


totalset=pd.DataFrame(index=range(len(historicalsale)))
createdataset(totalset, historicalsale, masterdata, commercialplanning,Twotemps ,5, 6)


# In[ ]:





# In[47]:





# In[ ]:





# In[ ]:





# In[ ]:




