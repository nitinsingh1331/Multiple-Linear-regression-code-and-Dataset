#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION 

# In[ ]:





# # IMPORT LIBRARIES 

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
df=pd.read_csv("C:/Users/Dell/Downloads/50_Startups.csv")
df 


# In[2]:


x=df.iloc[:,:-1] 
y=df.iloc[:,-1] 
y 


# In[3]:


sns.heatmap(df.corr()) 


# In[4]:


from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough") 
x=np.array(ct.fit_transform(x)) 
x 


# In[5]:


x=x[:,1:] 


# In[6]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 


# In[7]:


from sklearn.linear_model import LinearRegression 
regressor=LinearRegression() 
regressor.fit(X_train,y_train) 


# In[8]:


y_pred=regressor.predict(X_test) 
y_pred 


# In[9]:


y_test 


# # equation 
# 
# y=b0+b1x1....bnxn 

# In[10]:


regressor.coef_ 


# In[11]:


regressor.intercept_ 


# In[12]:


from sklearn.metrics import r2_score 
r2_score(y_test,y_pred )


# In[13]:


import statsmodels.formula.api as sm  
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) 


# # backward elimination 

# In[ ]:


x_op=x[:,[0,1,2,3,4,5]] 


# In[18]:


import statsmodels.api as sm 
OLS=sm.OLS(endog=y,exog=x_op).fit()
OLS.summary()


# In[19]:


x_op=x[:,[0,1,2,3,4]] 
import statsmodels.api as sm 
OLS=sm.OLS(endog=y,exog=x_op).fit()
OLS.summary()


# In[20]:


x_op=x[:,[0,1,2,3]] 
import statsmodels.api as sm 
OLS=sm.OLS(endog=y,exog=x_op).fit()
OLS.summary()


# In[21]:


x_op=x[:,[0,3]] 
import statsmodels.api as sm 
OLS=sm.OLS(endog=y,exog=x_op).fit()
OLS.summary()

