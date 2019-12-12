#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:52:15 2019
@author: axel
convert matrice to a kind of bed2d file:
chr1	0	2000	chr1	0	2000	78  
"""

import numpy as np
import scipy
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix 
import time
import itertools
import matplotlib.pylab as plt

bin_size = 500

# if dense:
mat = coo_matrix(D)

# if sparse: 
mat_chro = scipy.sparse.load_npz("/media/axel/RSG4/ChIA-PET/GM12878/500_SRR2312566_ChiaPET_CTCF/epstein_epstein_sparse_matrice2_500.txt.npz")
mat_chro = mat_chro.tocoo()

name="epstein"
# writting into a kind og bed2d file: 
f_out  = open('contacts2_'+name+"_"+str(bin_size)+'_df.bg2',"w+")
len(mat.data)
c1="chr0"
c2="chr0"

v_c1 = [c1] * len( mat_chro.data  )
vx1=mat_chro.row * bin_size
vy1=mat_chro.row * bin_size + bin_size
    
v_c2 = [c2] * len( mat_chro.data  )
vx2= mat_chro.col * bin_size
vy2= mat_chro.col * bin_size + bin_size
    
v_data = mat_chro.data
    
np.savetxt(f_out, np.c_[v_c1, list(vx1), list(vy1) ,
                    v_c2, list(vx2), list(vy2), 
                    v_data], fmt='%s', delimiter ="\t")
