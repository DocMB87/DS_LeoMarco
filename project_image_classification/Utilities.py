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

xpix = 299
ypix = 299
white_th = 20
black_th = 235
xsplit = 3
ysplit = 3
select = [0,2,4,6,8]
suffix = ['_'+i for i in string.ascii_uppercase] 

def RGBtoHSV_C(img):
    img = img / 255.0

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    m, M = np.min(img[:,:,:3], 2), np.max(img[:,:,:3], 2)
    d = M - m

    # Chroma 
    c = d    
    # Value (Brightness)
    v = M
    # Hue
    h = np.select([c ==0, r == M, g == M, b == M], [0, ((g - b) / c) % 6, (2 + ((b - r) / c)), (4 + ((r - g) / c))], default=0) * 60
    # Saturation (HSV)
    s = np.select([c == 0, c != 0], [0, c/v])
  
    # Lightness
    l=0.5*(v*(2-s))
    
    # Saturation (HSL)
    s_l=v*s/(1-np.abs(2*l-1))
    
    return (h,s,v,c,l,s_l)

def hash_RGB(img_RGB,color,suff,nimg):
    
    df_hash = pd.DataFrame(index=[nimg])
    
    for i in range(0,len(img_RGB)):
        cpil = Image.fromarray(img_RGB[i])
        cphash = imagehash.phash(cpil)
        cahash = imagehash.average_hash(cpil)
        cdhash = imagehash.dhash(cpil)

        df_hash.loc[nimg,'phash'+color[i]+suff] = str(cphash)
        df_hash.loc[nimg,'ahash'+color[i]+suff] = str(cahash)
        df_hash.loc[nimg,'dhash'+color[i]+suff] = str(cdhash)
    
    return df_hash

def func_fit(x, a, b):
    return a*x + b

def feature_eng_dict(img, suff):
    
    dict_func = {}
    
    height, width, channels = img.shape

    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # ncolor img
    img_flat = np.reshape(img, (height*width, 3))
    num_col = sum(np.unique(img_flat))
    dict_func['ncolor'+suff] = num_col

    # color balancing img
    R_G_diff = sum(sum(red))-sum(sum(green))
    R_B_diff = sum(sum(red))-sum(sum(blue))
    G_B_diff = sum(sum(green))-sum(sum(blue))

    dict_func['R_G_diff'+suff] = R_G_diff
    dict_func['R_B_diff'+suff] = R_B_diff
    dict_func['G_B_diff'+suff] = G_B_diff

    # BW img
    tot_px=height*width    # number of pixels
    white_perc = np.sum(np.sum(np.sum(img > white_th,axis=2)==3))/tot_px/3.0
    black_perc = np.sum(np.sum(np.sum(img < black_th,axis=2)==3))/tot_px/3.0
    
    dict_func['white_perc'+suff] = white_perc
    dict_func['black_perc'+suff] = black_perc

    # gray img
    cond = (img<=black_th) & (img>=white_th)
    gray_perc = np.sum(np.sum(np.sum(cond,axis=2)==3))/tot_px/3.0

    dict_func['gray_perc'+suff] = gray_perc

    # HSV, HSL img
    h,s,v,c,l,s_l=RGBtoHSV_C(img)

    h_mean=np.nanmean(h)
    s_mean=np.nanmean(s)
    v_mean=np.nanmean(v)
    c_mean=np.nanmean(c)
    l_mean=np.nanmean(l)
    s_l_mean=np.nanmean(s_l)

    h_median=np.nanmedian(h)
    s_median=np.nanmedian(s)
    v_median=np.nanmedian(v)
    c_median=np.nanmedian(c)
    l_median=np.nanmedian(l)
    s_l_median=np.nanmedian(s_l)

    h_var=np.nanvar(h)
    s_var=np.nanvar(s)
    v_var=np.nanvar(v)
    c_var=np.nanvar(c)
    l_var=np.nanvar(l)
    s_l_var=np.nanvar(s_l)
    
    dict_func['h_mean'+suff] = h_mean
    dict_func['s_mean'+suff] = s_mean
    dict_func['v_mean'+suff] = v_mean
    dict_func['c_mean'+suff] = c_mean
    dict_func['l_mean'+suff] = l_mean
    dict_func['s_l_mean'+suff] = s_l_mean

    dict_func['h_median'+suff] = h_median
    dict_func['s_median'+suff] = s_median
    dict_func['v_median'+suff] = v_median
    dict_func['c_median'+suff] = c_median
    dict_func['l_median'+suff] = l_median
    dict_func['s_l_median'+suff] = s_l_median

    dict_func['h_var'+suff] = h_var
    dict_func['s_var'+suff] = s_var
    dict_func['v_var'+suff] = v_var
    dict_func['c_var'+suff] = c_var
    dict_func['l_var'+suff] = l_var
    dict_func['s_l_var'+suff] = s_l_var        
    
    biasx = np.zeros(shape=(width,height))
    biasy = np.zeros(shape=(width,height))
    biasx[:,:] = 20
    biasy[:,:] = -20
    
    distq_RG0 = 2.0*np.sum((red - green)**2)
    distq_RB0 = 2.0*np.sum((red - blue)**2)
    
    distq_RG1 = np.sum((red - green + biasy)**2) + np.sum((red - green - biasx)**2)
    distq_RB1 = np.sum((red - blue + biasy)**2) + np.sum((red - blue - biasx)**2)
    
    distq_RG2 = np.sum((red - green + biasx)**2) + np.sum((red - green - biasy)**2)
    distq_RB2 = np.sum((red - blue + biasx)**2) + np.sum((red - blue - biasy)**2)
        
    dict_func['distq_RG0'+suff] = distq_RG0
    dict_func['distq_RB0'+suff] = distq_RB0

    dict_func['distq_RG1'+suff] = distq_RG1
    dict_func['distq_RB1'+suff] = distq_RB1

    dict_func['distq_RG2'+suff] = distq_RG2
    dict_func['distq_RB2'+suff] = distq_RB2
    
    return dict_func

