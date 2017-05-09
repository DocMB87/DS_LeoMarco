
# coding: utf-8

# In[1]:

import xgboost as xgb
import os
from sklearn.utils import shuffle
import sys
from math import *
import numpy as np
import pandas as pd
import csv

import string
from PIL import Image
import imagehash
from matplotlib.pyplot import imshow
from os import listdir
from os.path import isfile, join
import scipy.optimize as optimization
import PIL
import sklearn


# In[2]:

df_11 = pd.read_csv('Unisalute_df_2011.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)
df_12 = pd.read_csv('Unisalute_df_2012.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)
df_13 = pd.read_csv('Unisalute_df_2013.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)
df_14 = pd.read_csv('Unisalute_df_2014.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)
df_15 = pd.read_csv('Unisalute_df_2015.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)
df_16 = pd.read_csv('Unisalute_df_2016.csv',encoding="ISO-8859-1",sep = ';',
                 keep_default_na=True,low_memory=False)


# In[5]:

# Predictors selection
select = [0,2,4,6,8]
suffix = ['_'+i for i in string.ascii_uppercase] 

feat = ['ncolor','R_G_diff','R_B_diff','G_B_diff','white_perc','black_perc','gray_perc',     'h_mean','s_mean','v_mean','c_mean','l_mean','s_l_mean',     'h_median','s_median','v_median','c_median','l_median','s_l_median',     'h_var','s_var','v_var','c_var','l_var','s_l_var',     'distq_RG0','distq_RB0','distq_RG1','distq_RB1','distq_RG2','distq_RB2']

cl = ['_TOT']+suffix[:len(select)]
X = ['fn','year']
for el in feat:
    c = [el + x  for x in cl]
    X.extend(c)


# In[6]:

df_11 = df_11[X]
df_12 = df_12[X]
df_13 = df_13[X]
df_14 = df_14[X]
df_15 = df_15[X]
df_16 = df_16[X]


# In[8]:

frames = [df_11, df_12, df_13, df_14, df_15, df_16]
df_full = pd.concat(frames,ignore_index =True)

del frames, df_11, df_12, df_13, df_14, df_15, df_16


# In[16]:

df_full.to_csv('df_raw_11_16_raw.csv', sep=';', index=False)


# In[ ]:




# In[17]:

# Predictors selection
select = [0,2,4,6,8]
suffix = ['_'+i for i in string.ascii_uppercase] 

feat = ['ncolor','R_G_diff','R_B_diff','G_B_diff','white_perc','black_perc','gray_perc',     'h_mean','s_mean','v_mean','c_mean','l_mean','s_l_mean',     'h_median','s_median','v_median','c_median','l_median','s_l_median',     'h_var','s_var','v_var','c_var','l_var','s_l_var',     'distq_RG0','distq_RB0','distq_RG1','distq_RB1','distq_RG2','distq_RB2']

cl = ['_TOT']+suffix[:len(select)]
X = []
for el in feat:
    c = [el + x  for x in cl]
    X.extend(c)

y = 'type' 


# In[18]:

X_df=df_full[X]
preserved_mapper = {'TEXT':2 , 'COLOR': 0, 'GRAY': 1}
xg_df = xgb.DMatrix(X_df.values)


# In[19]:

model = xgb.Booster({'nthread':7}) 
model.load_model('xgb_17k_final_prob.model')


# In[20]:

prediction = model.predict(xg_df)
predictions_type = np.argmax(prediction,axis=1)


# In[24]:

df_output = df_full[['fn','year']]
df_output = df_output.join(pd.DataFrame({'predict':predictions_type}))


# In[25]:

df_output.to_csv('df_predicted_11_16.csv', sep=';', index=False)

