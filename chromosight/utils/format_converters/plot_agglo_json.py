#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:58:51 2019
@author: axel KournaK
To plot results of quantification of pattern
Ex: python plot_agglo_json.py loops_quant.json loops_quant.txt  arima
"""
import numpy as np
import scipy
import time
import itertools
import matplotlib.pylab as plt
import json
import pandas as pd
import matplotlib.gridspec as gridspec
import sys 

json_file = sys.argv[1]     # json file 
df = pd.read_csv(sys.argv[2],header=0, delimiter="\t")   # txt file
name_bank = str(sys.argv[3])    # name 

score_mean = np.nanmean(df['score'])

with open(json_file) as json_file:
    data = json.load(json_file)

mat_sum = np.zeros((17,17))
i=0
ne=0
mat_total = []
for i in range(len(data)) :
    di=data[str(i)]
    di=np.array(di)
    di[np.isinf(di)] = 1.0
    mat_total.append(di)   
    di[ di> 10] = 10 
    ne+=1
    mat_sum = mat_sum + di
    
#    if len(di[ di> 15] ) == 0 : # if no elements are crasy
#        ne+=1
#        mat_sum = mat_sum + di

print("Number of sub-matrices at initial:")
print(str(i) )
print("Number of sub-matrices summed:")
print(str(ne) )

plt.figure(1)
plt.imshow( mat_sum/ne , cmap="seismic", vmin=0., vmax=2.0)
plt.title("N= "  + str( len(data))+"\n"+"Mean=" + str( np.round(score_mean,2)))
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.savefig("agglo_json_"+name_bank+".pdf", dpi=400, format='pdf')
plt.close("all")

plt.figure(2)
plt.imshow( np.log(mat_sum/ne ), cmap="seismic", vmin=-1.0, vmax=1.0)
plt.title("N= "  + str( len(data))+"\n"+"Mean=" + str( np.round(score_mean,2)))
plt.colorbar()

plt.xticks([])
plt.yticks([])

plt.savefig("agglo_json_"+name_bank+"_log.pdf", dpi=400, format='pdf')

plt.close("all")

plt.figure(3)
plt.imshow( np.log(mat_sum/ne ), cmap="seismic", vmin=-0.2, vmax=0.2)
plt.title("N= "  + str( len(data))+"\n"+"M=" + str( np.round(score_mean,2)))
plt.colorbar()

plt.xticks([])
plt.yticks([])

plt.savefig("agglo_json_"+name_bank+"_log02.pdf", dpi=400, format='pdf')
plt.close("all")


# Scores files
s1=df

len(s1["score"])
scores1 = s1["score"] 
scores1 = scores1[~np.isnan(scores1)]

print(len(scores1))
print(min(scores1))
print(max(scores1))

# Histogram and plots:
h1 = plt.hist(scores1,200, range=[-1.0,1.0] )
plt.xlim(-1.0,1.0)
plt.xlabel("Pattern Score")
plt.ylabel("No. occurences")
plt.title("<Mean>~ "+str(float(np.mean(scores1)))+"\n"+"<Median> ~"+str(float(np.median(scores1))) )

plt.savefig("distrib_Loop_scores"+name_bank+".pdf", dpi=400, format='pdf')
plt.close('all')




