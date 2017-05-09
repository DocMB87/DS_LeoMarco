
# coding: utf-8

# In[1]:

import os
import sys
from math import *
import numpy as np
import pandas as pd
import time
import csv
from scipy.stats.stats import pearsonr   
from scipy.stats.stats import spearmanr 
import scipy
import string
import imagehash
from os import listdir
from os.path import isfile, join
import scipy.optimize as optimization
import PIL
import warnings
from joblib import Parallel, delayed
from PIL import Image
from Utilities import *

os.environ["http_proxy"] = "http://E3850004:Unipol66@proxybo.servizi.gr-u.it:80"
os.environ["https_proxy"] = "http://E3850004:Unipol66@proxybo.servizi.gr-u.it:80"


# In[2]:

# define global variables
xpix = 299
ypix = 299
white_th = 20
black_th = 235
xsplit = 3
ysplit = 3
select = [0,2,4,6,8]
suffix = ['_'+i for i in string.ascii_uppercase] 

# define path
fol_main = './'
year = ['2016']
fol = ['']
fol[0] = fol_main+year[0]+os.sep

ifol = 0   

included_extensions = ['jpg', 'jpeg', 'tif']
file = [fn for fn in listdir(fol[ifol]) if any(fn.endswith(ext) for ext in included_extensions)]

print('\nfolder : ',year[ifol],'\n', 'n_file', len(file))

# Ignore runtime errors (nan means etc)
warnings.simplefilter("ignore", category=RuntimeWarning)


# In[3]:

def par_job_dict(ifil): 
    dict_temp = {}
    
    img = Image.open(fol[ifol]+ifil)
    img = img.resize((xpix,ypix), PIL.Image.ANTIALIAS)
    img=img.convert('RGB')
    img = np.array(img) 
    
    dict_temp['fn'] = str(ifil)
    dict_temp['year'] = year[ifol]    

    dict_func = feature_eng_dict(img, '_TOT')
    dict_temp.update(dict_func)
    
    # divide img in subimg
    height, width, channels = img.shape
    xticks = [round(i*1.0/xsplit*width) for i in range(xsplit)] 
    xticks.append(width)
    yticks = [round(i*1.0/xsplit*height) for i in range(ysplit)] 
    yticks.append(height)
    subimg = list(range(len(select)))

    for isel in range(0,len(select)):
        row = int((select[isel])/xsplit)
        col = (select[isel])%xsplit
        subimg[isel] = img[xticks[row]:xticks[row+1],yticks[col]:yticks[col+1]]

        dict_func = feature_eng_dict(subimg[isel], suffix[isel])

        dict_temp.update(dict_func)

    return dict_temp


# In[ ]:

num_cores = 8

out = Parallel(n_jobs=num_cores, backend = 'multiprocessing', pre_dispatch = '3*n_jobs', verbose=1, 
                   batch_size = 1, max_nbytes='10G',)(delayed(par_job_dict)(ifil) for ifil in file)

t0 = time.time()
df = pd.DataFrame(out)
time.time() -t0 


# In[ ]:

# Predictors selection
select = [0,2,4,6,8]
suffix = ['_'+i for i in string.ascii_uppercase] 

feat = ['ncolor','R_G_diff','R_B_diff','G_B_diff','white_perc','black_perc','gray_perc',     'h_mean','s_mean','v_mean','c_mean','l_mean','s_l_mean',     'h_median','s_median','v_median','c_median','l_median','s_l_median',     'h_var','s_var','v_var','c_var','l_var','s_l_var',     'distq_RG0','distq_RB0','distq_RG1','distq_RB1','distq_RG2','distq_RB2']

cl = ['_TOT']+suffix[:len(select)]
X = ['fn','year']
for el in feat:
    c = [el + x  for x in cl]
    X.extend(c)


# In[ ]:

df = df[X]


# In[ ]:

df.to_csv(path_or_buf='Unisalute_df_'+ str(year[0]) +'.csv', sep=';', index=False)


# In[ ]:



