#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:51:19 2019
@author: axel KournaK
To convert pairs files into bed2d format 
ex: 
chr1	0	1000	chr1	0	1000	118
chr1	0	1000	chr1	1000	2000	164    
"""
import numpy as np
import scipy
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix 
import time
import itertools
import matplotlib.pylab as plt
import sys
import pandas as pd 

# Parameters to enter:
filename1 = sys.argv[1]

#list_all_chrms=('NC_003421.2','NC_003423.3','NC_003424.3')

bin_size = 2000  # size of the bin for the sparse matrices 
N=120000     # maximal size of sparse matrice 
chunksize = 1000000 # number of lines put into dict, to adapt for RAM consumption 
option_inter = 1  #  option to choose or not to computer inter-chrm sparse matrices 

# To determine list of chromosomes
i=0
j=0
list_chr=[]
with open(filename1) as f: # open the file for reading output alignment file
    i=0
    for line in f: # iterate over each line
        i+=1
        chr1, locus1, sens1,indice1, chr2, locus2, sens2,indice2 = line.split()  # split it by whitespace
        list_chr.append(chr1)
        list_chr.append(chr2)
        if i==1000000 :
            break
        
list_all_chrms = np.unique(list_chr)
# 
k=0
f_num = {}
for c in list_all_chrms :
    k+=1
    f_num[c] = k

list_combi_chrs = list(itertools.combinations_with_replacement(list_all_chrms,2) )

mat={}
mat_chro = {}
for c1, c2 in list_combi_chrs :
    mat[c1, c2]= {}  # temporary dictionary object to keep in memory the contacts (erased every chunk)
    mat_chro[c1, c2] = coo_matrix((N, N), dtype=np.int8) # sparse object 

start = time.time()
i=0
j=0
with open(filename1) as f: # open the file for reading output alignment file
    i=0
    for line in f: # iterate over each line
        i+=1
        chr1, locus1, sens1,indice1, chr2, locus2, sens2,indice2 = line.split()  # split it by whitespace
        locus1=int(locus1)
        locus2=int(locus2)
        
        if chr1 == chr2 and chr1 in list_all_chrms :
            if locus1 > locus2 :
                locus = locus1; locus1 = locus2; locus2=locus
            bin1 = int(locus1 /  bin_size)
            bin2 = int(locus2 /  bin_size)
            key1=(bin1, bin2) 
            if key1 in mat[chr1,chr2]:
                mat[chr1,chr2][key1] += 1
            else:
                mat[chr1,chr2][key1] = 1
                
        if chr1 != chr2 and chr1 in list_all_chrms and chr2 in list_all_chrms and option_inter == 1:
            if f_num[chr1] > f_num[chr2] :  # we inverse to have always the same order
                chr0 = chr1; chr1=chr2; chr2=chr0 
                locus = locus1; locus1 = locus2; locus2=locus
            bin1 = int(locus1 /  bin_size)
            bin2 = int(locus2 /  bin_size)
            key1=(bin1, bin2) 
            if key1 in mat[chr1,chr2]:
                mat[chr1,chr2][key1] += 1
            else:
                mat[chr1,chr2][key1] = 1
        # update here: conversion of the dict object into sparses matrices :        
        if i % chunksize == 0  :
            print(str(i)+" lines parsed.")
            
            if option_inter == 1 :
                for c1, c2 in list_combi_chrs:
                    list_x = []
                    list_y = []
                    list_v = list(mat[c1, c2].values() )
                    for k, v in mat[c1, c2].items() : 
                        list_x.append(k[0])
                        list_y.append(k[1])
                    mat_chro[c1,c2] = mat_chro[c1,c2] + coo_matrix((list_v, (list_x, list_y)), shape=(N, N) )
                    
                mat.clear()
                for c1, c2 in list_combi_chrs :
                    mat[c1, c2]= {}  # dictionary object to keep in memory the contacts (erased every chunk)
                list_x.clear()
                list_y.clear()
                list_v.clear()
            else :    # intra only 
                for c1, c2 in zip(list_all_chrms, list_all_chrms):
                    list_x = []
                    list_y = []
                    list_v = list(mat[c1, c2].values() )
                    for k, v in mat[c1, c2].items() : 
                        list_x.append(k[0])
                        list_y.append(k[1])     
                    mat_chro[c1,c2] = mat_chro[c1,c2] + coo_matrix((list_v, (list_x, list_y)), shape=(N, N) )
                    
                mat.clear()
                for c1, c2 in list_combi_chrs :
                    mat[c1, c2]= {} # dictionary object to keep in memory the contacts (erased every chunk)
                list_x.clear()
                list_y.clear()
                list_v.clear()
    else :  # process the last part of the file:
        print(str(i)+" lines parsed.")
        if option_inter == 1 :
            for c1, c2 in list_combi_chrs:
                list_x = []
                list_y = []
                list_v = list(mat[c1, c2].values() )
                for k, v in mat[c1, c2].items() : 
                    list_x.append(k[0])
                    list_y.append(k[1])
                mat_chro[c1,c2] = mat_chro[c1,c2] + coo_matrix((list_v, (list_x, list_y)), shape=(N, N) )
                
            mat.clear()
            for c1, c2 in list_combi_chrs :
                mat[c1, c2]= {}  # dictionary object to keep in memory the contacts (erased every chunk)
            list_x.clear()
            list_y.clear()
            list_v.clear()
        else :    # intra only 
            for c1, c2 in zip(list_all_chrms, list_all_chrms):
                list_x = []
                list_y = []
                list_v = list(mat[c1, c2].values() )
                for k, v in mat[c1, c2].items() : 
                    list_x.append(k[0])
                    list_y.append(k[1])     
                mat_chro[c1,c2] = mat_chro[c1,c2] + coo_matrix((list_v, (list_x, list_y)), shape=(N, N) )
                
            mat.clear()
            for c1, c2 in list_combi_chrs :
                mat[c1, c2]= {} # dictionary object to keep in memory the contacts (erased every chunk)
            list_x.clear()
            list_y.clear()
            list_v.clear()
            
end = time.time()
print(end-start)


# Reshape and saving of sparse objects:
for c1, c2 in mat_chro.keys():
    mat_chro[c1,c2] = coo_matrix(mat_chro[c1,c2])
    if len(mat_chro[c1,c2].data) > 0 :
        nx = max(mat_chro[c1,c2].row) + 1
        ny = max(mat_chro[c1,c2].col) + 1   
        if c1 == c2:
            nx = max(nx, ny)  
            ny = nx
        mat_chro[c1,c2] = mat_chro[c1,c2].tocsr()    
        csr_matrix.resize(mat_chro[c1,c2], (nx,ny) )    
        scipy.sparse.save_npz(c1+"_"+c2+"_sparse_matrice_"+str(bin_size)+".txt", 
                              mat_chro[c1,c2], compressed=True)
    else : 
        nx = 0
        ny = 0
        mat_chro[c1,c2] = mat_chro[c1,c2].tocsr()  
        csr_matrix.resize(mat_chro[c1,c2], (nx,ny) )
        scipy.sparse.save_npz(c1+"_"+c2+"_sparse_matrice_"+str(bin_size)+".txt", 
                              mat_chro[c1,c2], compressed=True)
        
             
# writting into bed 2D file:   
#    chr1	0	1000	chr1	0	1000	118
f_out  = open('contacts_df.bg2',"w+")  
      
for c1, c2 in mat_chro.keys() :
    mat_chro[c1,c2] = coo_matrix(mat_chro[c1,c2])
    
    v_c1 = [c1] * len( mat_chro[c1,c2].data  )
    vx1=mat_chro[c1,c2].row * bin_size
    vy1=mat_chro[c1,c2].row * bin_size + bin_size
    
    v_c2 = [c2] * len( mat_chro[c1,c2].data  )
    vx2= mat_chro[c1,c2].col * bin_size
    vy2= mat_chro[c1,c2].col * bin_size + bin_size
    
    v_data = mat_chro[c1,c2].data
    
    np.savetxt(f_out, np.c_[v_c1, list(vx1), list(vy1) ,
                       v_c2, list(vx2), list(vy2), 
                       v_data], fmt='%s', delimiter ="\t")

f_out.close()
        
        
        
        
        
        
