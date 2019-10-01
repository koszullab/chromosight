#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:51:19 2019
@author: axel KournaK
To convert pairs files into sparse object for each chrm 
and each inter chrm submatrice
less greedy in RAM ~ 4 Gb
"""
import numpy as np
import scipy
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix 
import time
import sys
import itertools
import matplotlib.pylab as plt

# Parameters to enter:
filename1 = "/home/axel/Bureau/Hela/SRR5952305_Hi-Ccontrol.pairs.myformat.indices.filtered.100000000"

BIN = 10000  # size of the bin for the sparse matrices 
N=25000     # maximal size of sparse matrice 
chunksize = 1000000  # number of lines put into dict, to adapt for RAM consumption 
option_inter = 1  #  option to choose or not to computer inter-chrm sparse matrices 

list_all_chrms=('chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9'
                ,'chr10','chr11','chr12','chr13','chr14','chr15','chr16',
                'chr17','chr18','chr19','chr20','chr21','chr22','chrX','chrY',
                'chrM')
# 
k=0
f_num = {}
for c in list_all_chrms :
    k+=1
    f_num[c] = k

list_combi_chrs = list(itertools.combinations_with_replacement(list_all_chrms,2) )

mat={}
MAT_CHRO = {}
for c1, c2 in list_combi_chrs :
    mat[c1, c2]={}  # temporary dictionary object to keep in memory the contacts (erased every chunk)
    MAT_CHRO[c1, c2] = coo_matrix((N, N), dtype=np.int8) # sparse object 

start = time.time()
i=0
j=0
with open(filename1) as f: # open the file for reading output alignment file
    i=0
    for line in f: # iterate over each line
        i+=1
        chr1, locus1, sens1,indice1, chr2, locus2, sens2,indice2 = line.split()  # split it by whitespace
        locus1=int(locus1);sens1=int(sens1);
        locus2=int(locus2);sens2=int(sens2); 
        if chr1 == chr2 and chr1 in list_all_chrms :
            if locus1 > locus2 :
                locus = locus1; locus1 = locus2; locus2=locus
            bin1 = int(locus1 /  BIN)
            bin2 = int(locus2 /  BIN)
            key1=(bin1, bin2) 
            if key1 in mat[chr1,chr2]:
                mat[chr1,chr2][key1] += 1
            else:
                mat[chr1,chr2][key1] = 1
        if chr1 != chr2 and chr1 in list_all_chrms and chr2 in list_all_chrms and option_inter == 1:
            if f_num[chr1] > f_num[chr2] :  # we inverse to have always the same order
                chr0 = chr1; chr1=chr2; chr2=chr0 
                locus = locus1; locus1 = locus2; locus2=locus
            bin1 = int(locus1 /  BIN)
            bin2 = int(locus2 /  BIN)
            key1=(bin1, bin2) 
            if key1 in mat[chr1,chr2]:
                mat[chr1,chr2][key1] += 1
            else:
                mat[chr1,chr2][key1] = 1        
        # update here: conversion of the dict object into sparses matrices :        
        if i % chunksize == 0 :
            print(str(i)+" lines parsed.") 
            if option_inter == 1 :
                for c1, c2 in list_combi_chrs:
                    list_x = []
                    list_y = []
                    list_values = []
                    for k, v in mat[c1, c2].items() : 
                        list_x.append(k[0])
                        list_y.append(k[1])
                        list_values.append(v)
                    mat_chro = coo_matrix((list_values, (list_x, list_y)), shape=(N, N) )
                    MAT_CHRO[c1,c2] = MAT_CHRO[c1,c2]  + mat_chro
                    
                mat.clear()
                for c1, c2 in list_combi_chrs :
                    mat[c1, c2]={}  # dictionary object to keep in memory the contacts (erased every chunk)
                list_x.clear()
                list_y.clear()
                list_values.clear()
            else :
                for c1, c2 in zip(list_all_chrms, list_all_chrms):
                    list_x = []
                    list_y = []
                    list_values = []
                    for k, v in mat[c1, c2].items() : 
                        list_x.append(k[0])
                        list_y.append(k[1])
                        list_values.append(v)     
                    mat_chro = coo_matrix((list_values, (list_x, list_y)), shape=(N, N) )
                    MAT_CHRO[c1,c2] = MAT_CHRO[c1,c2]  + mat_chro 
                    
                mat.clear()
                for c1, c2 in list_combi_chrs :
                    mat[c1, c2]={}  # dictionary object to keep in memory the contacts (erased every chunk)
                list_x.clear()
                list_y.clear()
                list_values.clear()

end = time.time()
print(end-start)

# Reshape and saving:
for c1, c2 in list_combi_chrs:
    MAT_CHRO[c1,c2] = coo_matrix(MAT_CHRO[c1,c2])
    if len(MAT_CHRO[c1,c2].data) > 0 :
        Nx = max(MAT_CHRO[c1,c2].row)
        Ny = max(MAT_CHRO[c1,c2].col)    
        MAT_CHRO[c1,c2] = MAT_CHRO[c1,c2].tocsr()
        csr_matrix.resize(MAT_CHRO[c1,c2], (Nx,Ny) )
        scipy.sparse.save_npz(c1+"_"+c2+"_sparse_matrice_"+str(BIN)+".txt", 
                              MAT_CHRO[c1,c2], compressed=True)
        
# Test and plot:

sparse_matrix = scipy.sparse.load_npz('/home/axel/Bureau/Hela/sparse_objects/chr2_chr3_sparse_matrice_10000.txt.npz')
plt.spy(sparse_matrix,  markersize=0.005)

