#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:10:32 2019
@author: axel KournaK
To compare the distribution of scores given by ChromoSight
"""
import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.stats

# Input: take two quant files: 
df1=pd.read_table('/home/axel/Bureau/test_chromosight/yeast/SRR8769554_stop-alpha-factor-G1/loops_quant_cohe_classic_10_50kb_G1.txt',header=0, delimiter="\t") 
bank1="G1"
df2=pd.read_table('/home/axel/Bureau/test_chromosight/yeast/SRR8769549_mitotic/loops_quant_cohe_classic_10-50kb_49_mitotic.txt',header=0, delimiter="\t") 
bank2="Mitotic"


# Removal on Nan:
df4 = [df1['score'] , df2['score']]
df4 = pd.DataFrame({'score1':  df1['score'], 
                    'score2': df2['score']}, columns=['score1', 'score2'])

df4 = df4[(np.isfinite(df4['score1']) & np.isfinite(df4['score2']))]

# Stastical test: 
statistic,  pvalue = scipy.stats.mannwhitneyu(df4['score1'], 
                                              df4['score2'], 
                                              alternative='two-sided')

#  Plots: 
# Bo
## combine these different collections into a list    
data_to_plot = [df4['score1'], df4['score2']]

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True)

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    #box.set( facecolor = '#1b9e77' )
    box.set( facecolor = 'gold' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    #median.set(color='#b2df8a', linewidth=2)
    median.set(color='red', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)


ax.set_xticklabels([bank1, bank2])
ax.set_ylabel('Loops Score')
ax.set_title("Median1~"+str(round(np.median(df4['score1']),3))+
             ", Median2~"+str(round(np.median(df4['score2']),3))+
             ", Nb. elements="+str(len(df4)))
ax.set_ylabel('Loops Score')
ax.set_xlabel('Paired Mann Withney test - p-value~'+str( format(pvalue, '.2g') ) )

plt.savefig("box_plot_"+bank1+"_"+bank2+".pdf", dpi=500, format='pdf')
plt.close("all")


## Violin plot:
sns.set(style="white")

ax = sns.violinplot(data=df4)
ax.set_xticklabels([bank1, bank2])
ax.set_ylabel('Loops Score')
ax.set_title("Median1~"+str(round(np.median(df4['score1']),3))+
             ", Median2~"+str(round(np.median(df4['score2']),3))+
             ", Nb. elements="+str(len(df4)))
ax.set_ylabel('Loops Score')
ax.set_xlabel('Paired Mann Withney test, p-value~'+str( format(pvalue, '.2g') ) )

plt.savefig("violin_plot_"+bank1+"_"+bank2+".pdf", dpi=500, format='pdf')
plt.close("all")

