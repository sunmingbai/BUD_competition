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


# In[13]:


historicalsale


# In[14]:


historicalsale['Date']=historicalsale['Date'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d'))


# In[15]:


masterdata


# In[16]:


commercialplanning


# In[19]:


# load temperature


# In[17]:


def createdataset(totalset, historicalsale, masterdata, planning, rolling, lagging):
    totalset[['Year','Month','SalesRegion','SKU Code']]=historicalsale[['Year','Month','SalesRegion','SKU Code']]
    totalset['Weekday']=historicalsale['Date'].map(lambda x:x.weekday())
    # add temperaature and other time series related features
    # temperature


# In[18]:


totalset=pd.DataFrame(index=range(len(historicalsale)))
createdataset(totalset, historicalsale, masterdata, commercialplanning, 5, 6)
totalset


# In[ ]:




