#!/usr/bin/env python
# coding: utf-8

# In[50]:


import yfinance as yf
import numpy as np


# In[51]:


df=yf.download('INFY',start='2010-01-01')


# In[52]:


df


# In[53]:


df['returns']=np.log(df.Close.pct_change()+1)


# In[54]:


def lagit(df,lags):
    names=[]
    for i in range(1,lags+1):
        df['Lag_'+str(i)]=df['returns'].shift(i)
        names.append('Lag_'+str(i))
    return names


# In[55]:


lagnames = lagit(df,5)


# In[56]:


df


# In[57]:


df.dropna(inplace=True)


# In[58]:


from sklearn.linear_model import LinearRegression


# In[59]:


model=LinearRegression()


# In[60]:


model.fit(df[lagnames], df['returns'])


# In[61]:


df['prediction_LR']=model.predict(df[lagnames])


# In[62]:


df['direction_LR']=[1 if i>0 else -1 for i in df.prediction_LR]


# In[63]:


df['strat_LR']=df['direction_LR']* df['returns']


# In[64]:


np.exp(df[['returns','strat_LR']].sum())


# In[65]:


np.exp(df[['returns','strat_LR']].cumsum()).plot()


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


train,test=train_test_split(df, shuffle=False,test_size=0.3, random_state=0)


# In[68]:


train = train.copy()


# In[69]:


test=test.copy()


# In[70]:


model=LinearRegression()


# In[71]:


train


# In[72]:


test


# In[73]:


model.fit(train[lagnames],train['returns'])


# In[74]:


test['prediction_LR']=model.predict(test[lagnames])


# In[75]:


test['direction_LR']=[1 if i>0 else -1 for i in test.prediction_LR]


# In[76]:


test['strat_LR']=test['direction_LR']*test['returns']


# In[77]:


np.exp(test[['returns','strat_LR']].sum())


# In[78]:


(test['direction_LR'].diff() !=0).value_counts()


# In[79]:


np.exp(test[['returns','strat_LR']].cumsum()).plot()


# In[ ]:




