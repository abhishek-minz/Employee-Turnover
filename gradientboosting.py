#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import asarray
from numpy import mean
from numpy import std
from datetime import datetime as dt


# In[2]:


data = pd.read_csv('HR_Data.csv')


# In[3]:


print("\nSample Data")
data.head()


# In[4]:


data.dtypes


# In[5]:


#Checking for missing values
data.isnull().any()


# In[6]:


data.shape


# ### Data Exploration

# In[7]:


data['left'].value_counts()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
pd.crosstab(data.Department,data.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')


# In[9]:


table=pd.crosstab(data.salary, data.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


# In[10]:


num_bins = 10
data.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()


# In[11]:


cat_vars=['Department','salary']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1


# In[12]:


data.drop(data.columns[[8, 9]], axis=1, inplace=True)
data.columns.values


# In[13]:


data_vars=data.columns.values.tolist()
y=['left']
X=[i for i in data_vars if i not in y]


# In[14]:


X


# In[15]:


# independant variables
X = data.drop(['left'], axis=1)
# the dependent variable
y = data[['left']]


# ### XGBoost

# In[16]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[17]:


# Split X and y into training and test set in 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[18]:


# evaluate the model
model1 = XGBClassifier()
start = dt.now()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
running_secs = (dt.now() - start).seconds
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print('Time: %.3f' % running_secs)


# In[19]:


# fit the model on the whole dataset
model1 = XGBClassifier()
model1.fit(X, y)


# ### lightGBM

# In[20]:


from lightgbm import LGBMClassifier


# In[21]:


# evaluate the model
model2 = LGBMClassifier()
start = dt.now()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model2, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
running_secs = (dt.now() - start).seconds
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print('Time: %.3f' % running_secs)


# In[22]:


# fit the model on the whole dataset
model2 = LGBMClassifier()
model2.fit(X, y)


# ### CatBoost

# In[23]:


from catboost import CatBoostClassifier


# In[24]:


model = CatBoostClassifier(verbose=0, n_estimators=100)


# In[25]:


# evaluate the model
model3 = CatBoostClassifier(verbose=0, n_estimators=100)
start = dt.now()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model3, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
running_secs = (dt.now() - start).seconds
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print('Time: %.3f' % running_secs)


# In[26]:


# fit the model on the whole dataset
model = CatBoostClassifier(verbose=0, n_estimators=100)
model.fit(X, y)

