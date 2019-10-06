#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:23:09 2019
@author: axel
convert bg2 to sparse objects, [1 million lines in ~1.5 sec]
"""
import numpy as np
import time
import pandas as pd
import os
import errno
import itertools
import scipy
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix 

filename1 = "/home/axel/Bureau/chromo_sparse/example.bg2.txt2" # input file
output_dir = "/home/axel/Bureau/compressed_files/"    # output directory 

if not os.path.exists(os.path.dirname(output_dir)):
    try:
        os.makedirs(os.path.dirname(output_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

N = 200000 # maxiamal size of all sparse matrices 
bin_size =  10000 # in bp, bin size of the binning 

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

mat_chro = {}
for c1, c2 in list_combi_chrs :
    mat_chro[c1, c2] = coo_matrix((N, N), dtype=np.int8) # sparse object 

# Loading of the data, only the necessary : 
chunksize = 10 ** 6

start = time.time()
i=0
j=0
for chunk in pd.read_csv(filename1, chunksize=chunksize, 
                         header=None, delimiter="\t",usecols=[0,1,3,4,6]):
    for c1 in  chunk[0].unique():
        for c2 in  chunk[3].unique():
            if [c1, c2] != sorted((c1,c2)):  # if we found lines with wrong chrom order 
                chunk_temp = chunk.loc[ ( (chunk[0] == c1) & (chunk[3] == c2) )]
                x_binned = [ int(x/bin_size) for x in  chunk_temp[4]]
                y_binned = [ int(x/bin_size) for x in  chunk_temp[1]]
                mat_chro[c2,c1] = mat_chro[c2,c1] + coo_matrix(
                        (chunk_temp[6],(x_binned, y_binned)) , shape=(N, N) )
            else :
                chunk_temp = chunk.loc[ ( (chunk[0] == c1) & (chunk[3] == c2) )]
                x_binned = [ int(x/bin_size) for x in  chunk_temp[1]]
                y_binned = [ int(x/bin_size) for x in  chunk_temp[4]]
                mat_chro[c1,c2] = mat_chro[c1,c2] + coo_matrix(
                        (chunk_temp[6],(x_binned, y_binned)) , shape=(N, N) )
                
end = time.time()
print(end-start)


# Reshape and saving:
list_chr1 =[]
list_chr2 =[]
list_path =[]

# Reshape and saving:
for c1, c2 in mat_chro.keys():
    mat_chro[c1,c2] = coo_matrix(mat_chro[c1,c2])
    if len(mat_chro[c1,c2].data) > 0 :
        nx = max(mat_chro[c1,c2].row) + 1
        ny = max(mat_chro[c1,c2].col) + 1   
        if c1 == c2:
            nx = max(nx, ny) + 1 
            ny = max(nx, ny) + 1
        mat_chro[c1,c2] = mat_chro[c1,c2].tocsr()     
        csr_matrix.resize(mat_chro[c1,c2], (nx,ny) )
        name_file = c1+"_"+c2+"_sparse_matrice_"+str(bin_size)+".txt"
        path= output_dir+name_file
        scipy.sparse.save_npz(name_file,mat_chro[c1,c2], compressed=True)
        list_chr1.append(c1)
        list_chr2.append(c2)
        list_path.append(path)
       
           
# dataframe with the paths:         
df = pd.DataFrame(
    {'chr1': list_chr1,
     'chr2': list_chr2,
     'path': list_path
    })

np.savetxt(r'file_dataframe.txt', df.values, fmt='%s')
