#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:15:46 2018
@author: axel KournaK 
Create all possible pairs from genomic intervals in a bed file
Outputs a bed2d file for chromosight mode. 
"""
import numpy as np
import pandas as pd
import itertools as it
import sys

def bed_combo(in_f, out_f, min_range=0, max_range=np.inf):
    """
    Given an input bed file, generate all possible pairs of
    intervals to generate a bed2d file. Possibility
    to record only combinations between regions closer or further
    than threshold distances.
    """
    # Input: list of genomic positions :
    df=pd.read_csv(in_f, header=None, sep="\t")

    out_handle = open(out_file, "w+")
    # Generate all intrachromosomal combinations
    for chrom in np.unique(df[0]):
        chrom_df = df.loc[df[0] == chrom, :]
        for i1, i2 in it.combinations(range(chrom_df.shape[0]),2):
            start1, end1 = chrom_df.iloc[i1, [1, 2]]
            start2, end2 = chrom_df.iloc[i2, [1, 2]]
            dist = np.abs(start2 - start1)
            if dist >= min_range and dist < max_range:
                out_handle.write(f'{chrom}\t{start1}\t{end1}\t{chrom}\t{start2}\t{end2}\n')
    out_handle.close()  

usage = f"""
Usage: {sys.argv[0]} input.bed output.bed2d mindist maxdist
Generate all intrachromosomal combinations between intervals in a bed file.

Mandatory:
    input.bed : Input bed file.
    output.bed2d : Ouptut bed2d file with combinations.
Optional:
    mindist : Minimum distance, in basepairs to register pairs [default: 0]
    maxdist : Maximum distance [default: inf]
"""

if len(sys.argv) < 3:
    print('Not enough arguments')
    print(usage)
    sys.exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]
try:
    mindist = int(sys.argv[3])
except IndexError:
    mindist = 0

try:
    maxdist = int(sys.argv[4])
except IndexError:
    maxdist = np.inf

try:
    bed_combo(in_file, out_file, min_range=mindist, max_range=maxdist)
except ValueError as e:
    print(usage)
    raise e

