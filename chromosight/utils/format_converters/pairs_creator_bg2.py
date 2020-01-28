# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:15:46 2018
@author: axel KournaK 
To create all possible pairs from genomic positions
The output will a bg2 file for notably ChromoSight mode. 
"""
import numpy as np
import matplotlib
from pylab import *
import pandas as pd
import itertools

# Input: list of genomic positions :
name="Gm12878Rad21V0416101UniPk"
df=pd.read_table('/home/axel/Bureau/human_bioanalysis/chipseq_GM12878/wgEncodeAwgTfbsHaibGm12878Rad21V0416101UniPk.narrowPeak.4000',header=None, delimiter="\t")
print(len(df))

#name="SMC_peaks_finder_PY79"
#df=pd.read_table('/media/axel/RSG4/bacillus_2018/bacillus_ChIP_data/chip_chip_SMC/SMC_peaks_finder_PY79.txt',header=None, delimiter=" ")
#print(len(df))

# Processing: we keep only chr pos 
df1 = df[[0,1]]
mean_pos = (df[1] + df[2]) /2.0
mean_pos =  [ int(x) for x in mean_pos ]
df1[1] = mean_pos 

# for bacillus:
#df1 = df

list_chr =  ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10",\
             "chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX"]
#------------------------------------------------------------------------------

f_out1 = open("pairs_intra_groupe_"+name+".txt","w+")
# Intra and inter pairs :
for chro in list_chr :
    print(chro)    
    len(df1)
    df11 = df1.loc[ ( df1[0] == chro)  ]
    list_pairs = list(itertools.combinations( range(len(df11)) ,2 ))
    print(len(list_pairs))     
    i_intra=0
    v_pos = list(df11[1])
    
    for e1, e2 in list_pairs :
        pos1 = v_pos[e1]
        pos2 = v_pos[e2]
        i_intra +=1
        f_out1.write(chro + '\t'+ str( int(pos1) )+ '\t' + str( int(pos1+BIN))+ '\t' + 
                     chro + '\t'+ str( int(pos2)) + '\t' + str( int(pos2+BIN))+ '\t' + "1" +'\n')

    print("Number of pairs in intra:")
    print(i_intra)

f_out1.close()  



  