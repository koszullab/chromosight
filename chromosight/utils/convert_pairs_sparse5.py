#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:51:19 2019
@author: axel KournaK
To convert pairs files into sparse object for each chrm 
and each inter chrm submatrice
less greedy in RAM ~ 1.5 Gb - ~4 sec per million of line 
"""
import numpy as np
import scipy
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix 
import time
import itertools
import matplotlib.pylab as plt
import hicstuff as hcs

# Parameters to enter:
filename1 = "/home/axel/Bureau/Hela/compressed_file/SRR5952305_Hi-Ccontrol.pairs.myformat.indices.filtered"
filename1 = "/media/axel/RSG4/RSG5_copy/Dixon/output_alignment_idpt.dat.indices.filtered.pcr4"
filename1 = "/media/axel/RSG4/bacillus_2018/out_files/output_alignment_idpt.SRR2214059.dat.indices.filtered"
filename1 = "/media/axel/d0a28364-6c64-4f8e-9efc-f332d9a0f1a9/Next_Seq_Pierrick_May_2018_run3_Aragon/output_alignment_idpt_BC172_CGGT_SUPT16H_AID_DMSO_1A.dat.indices.filtered.pcr5"
filename1 = "/home/axel/Bureau/Arima/SRR6675327_Arima_EBV.pairs.myformat.indices.filtered.sorted3.pcr_geno_pos3" # the best 
filename1 = "/home/axel/Bureau/Rao/SRR1658592_Rao_Gm12878_HiC20.pairs.myformat.indices.filtered.sorted.pcr_geno_pos3"

filename1 = "/media/axel/RSG4/diverse_yeast_data/quiescence_2019/output_alignment_DADE.WT_quiescence.dat.indices.filtered"
filename1 = "/media/axel/RSG4/diverse_yeast_data/quiescence_2019/output_alignment_DADE.WT_log.dat.indices.filtered"

bin_size = 2000  # size of the bin for the sparse matrices 
N=120000     # maximal size of sparse matrice 
chunksize = 1000000 # number of lines put into dict, to adapt for RAM consumption 
option_inter = 1  #  option to choose or not to computer inter-chrm sparse matrices 

list_all_chrms=('chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9'
                ,'chr10','chr11','chr12','chr13','chr14','chr15','chr16',
                'chr17','chr18','chr19','chr20','chr21','chr22','chrX','chrY',
                'chrM')


list_all_chrms=('chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9'
                ,'chr10','chr11','chr12','chr13','chr14','chr15','chr16',
                'chrM')

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
        
# Test and plot:
sparse_matrix = scipy.sparse.load_npz('/home/axel/Bureau/Hela/sparse_objects/chr1_chr1_sparse_matrice_10000.txt.npz')
plt.spy(sparse_matrix,  markersize=0.005)


sparse_matrix = scipy.sparse.load_npz('/media/axel/RSG4/diverse_yeast_data/quiescence_2019/sparse_objects_log/chrM_chrM_sparse_matrice_200.txt.npz')
plt.figure(1)
sparse_matrix = sparse_matrix + np.transpose(sparse_matrix)
mn = hcs.normalize_sparse(sparse_matrix)
plt.spy(mn,  markersize=0.01)

sparse_matrix = scipy.sparse.load_npz('/media/axel/RSG4/diverse_yeast_data/quiescence_2019/sparse_objects_quiescence/chrM_chrM_sparse_matrice_200.txt.npz')
sparse_matrix = sparse_matrix + np.transpose(sparse_matrix)
plt.figure(2)
mn = hcs.normalize_sparse(sparse_matrix)
plt.spy(mn,  markersize=0.01)



# for bacterial genomes: almost all elements are non zeros 
m =sparse_matrix 
m = m + np.transpose(m)
mn = hcs.normalize_sparse(m)
mn= mn.todense()
plt.imshow(np.power(mn,0.15) , interpolation = "none", cmap="afmhot_r")

mt=m+ np.transpose(m)
imshow(np.power(mt,0.15) , interpolation = "none", cmap="afmhot_r")

# computation of the size: 
a=sparse_matrix
print(a.data.nbytes + a.indptr.nbytes + a.indices.nbytes)



sparse_matrix = scipy.sparse.load_npz('/home/axel/chr1_chr1_sparse_matrice_2000.txt.npz')
