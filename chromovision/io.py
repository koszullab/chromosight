#!/usr/bin/env python3
# coding: utf-8
"""Input/Output related functions
Load and save contact matrices in sparse format
"""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def load_bedgraph2d(mat_path):
    """
    Loads a sparse Hi-C contact matrix in 2D bedgraph format into a sparse
    matrix and an array of chromosome start bins. Expects the file to have 7
    space-separated columns: chr1, start1, end1, chr2, start2, end2, contacts.

    Parameters
    ----------
    mat_path : str
        Path to the matrix in 2D bedgraph format.

    Returns
    -------
    mat: scipy.sparse.coo_matrix
        Matrix 
    chrom_start : list of numpy.array
        List of chromosome start bins.
    """
    bg2 = pd.read_csv(mat_path, delimiter=" ", header=None)
    bg2.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2", "contacts"]

    # estimate bin size from file
    bin_size = np.median(bg2.end1 - bg2.start1).astype(int)

    # Get number of bins per chromosome
    fragsA = bg2.loc[:, ["chr1", "end1"]].rename(columns={"chr1": "chr", "end1": "end"})
    fragsB = bg2.loc[:, ["chr2", "end2"]].rename(columns={"chr2": "chr", "end2": "end"})
    frags = pd.concat([fragsA, fragsB])
    chroms = frags.groupby("chr", sort=False).apply(lambda x: max(x.end))

    # Convert from BP to #bin
    chrom_start = chroms // bin_size

    # Shift by one to get starting bin, first one is zero
    chrom_start[1:] = chrom_start[:-1]
    chrom_start[0] = 0
    chrom_start = chrom_start.cumsum()

    # Get frags indices
    bg2["frag1"] = bg2.apply(lambda x: (x.start1 // bin_size) + chroms[x.chr1], axis=1)
    bg2["frag2"] = bg2.apply(lambda x: (x.start2 // bin_size) + chroms[x.chr2], axis=1)

    # Build sparse matrix from fragment indices
    n = chroms.sum() // bin_size
    mat = coo_matrix(
        (bg2.contacts, (bg2.frag1, bg2.frag2)), shape=(n, n), dtype=np.float64
    )

    # Get chroms into a 1D array of bin starts
    chrom_start = np.array(chrom_start.tolist())
    return mat, chrom_start


load_bedgraph2d("../data/cooler.mat.2bg")

