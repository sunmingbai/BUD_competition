#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# In[3]:


# Data preparation


# In[9]:


historicalsale=pd.read_csv("E:\工作\百威\(Replace)Historical Sales Volume 2016.1-2019.11.csv")
masterdata=pd.read_csv("E:\工作\百威\(Replace)Product Master Data.csv")
commercialplanning=pd.read_csv("E:\工作\百威\(Replace)Commercial Planning.csv")


# In[12]:


calendar=pd.read_csv("C:/Users/mings/BUD/rearranged_sale.csv")


# In[14]:


historicalsale['Date']=historicalsale['Date'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d'))


# In[23]:


historicalsale.set_index('Date',inplace=True)
historicalsale


# In[15]:


masterdata


# In[16]:


commercialplanning


# In[50]:


# load temperature
Htem=pd.read_csv("E:\工作\百威\Data\HWeather_t.csv")
Jtem=pd.read_csv("E:\工作\百威\Data\JWeather_t.csv")
#Htem.info()
Htem


# In[51]:


Htem['Temp_max']=Htem['Temp_max'].map(lambda x: re.match('(.*)℃',x)[1])
Htem['Temp_max']=Htem['Temp_max'].astype(int)
Htem['Temp_min']=Htem['Temp_min'].map(lambda x: re.match('(.*)℃',x)[1])
Htem['Temp_min']=Htem['Temp_min'].astype(int)


# In[52]:


Htem['Date']=Htem['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[53]:


Htem.info()


# In[54]:


Jtem['Temp_min']=Jtem['Temp_min'].map(lambda x: re.match('(.*)℃',x)[1])
Jtem['Temp_max']=Jtem['Temp_max'].map(lambda x: re.match('(.*)℃',x)[1])
Jtem['Temp_max']=Jtem['Temp_max'].astype(int)
Jtem['Temp_min']=Jtem['Temp_min'].astype(int)
Jtem['Date']=Jtem['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[55]:


Htem.info()


# In[24]:


def createdataset(totalset, historicalsale, masterdata, planning, rolling, lagging):
    totalset.index=historicalsale.index
    totalset[['Year','Month','SalesRegion','SKU Code']]=historicalsale[['Year','Month','SalesRegion','SKU Code']]
    totalset['Weekday']=totalset.index.map(lambda x:x.weekday())
    # add temperaature and other time series related features
    # temperature corresponding to date and region


# In[25]:


totalset=pd.DataFrame(index=range(len(historicalsale)))
createdataset(totalset, historicalsale, masterdata, commercialplanning, 5, 6)
totalset


# In[ ]:




