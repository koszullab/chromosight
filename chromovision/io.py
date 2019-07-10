#!/usr/bin/env python3
# coding: utf-8
"""Input/Output related functions
Load and save contact matrices in sparse format
"""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix, csr_matrix


def load_bedgraph2d(mat_path):
    """
    Loads a sparse Hi-C contact matrix in 2D bedgraph format into a sparse
    matrix and an array of chromosome start bins. Expects the file to have 7
    space-separated columns: chr1, start1, end1, chr2, start2, end2, contacts.
    Expects the file to contain same-size fragments (and not restriction fragments)

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
    bg2 = pd.read_csv(mat_path, delimiter="\t", header=None)
    bg2.head()
    bg2.columns = [
        "chr1",
        "start1",
        "end1",
        "chr2",
        "start2",
        "end2",
        "contacts",
    ]

    # estimate bin size from file
    bin_size = np.median(bg2.end1 - bg2.start1).astype(int)

    # Convert from BP to #bin
    bg2["start1"] = bg2["start1"] // bin_size
    bg2["end1"] = bg2["end1"] // bin_size
    bg2["start2"] = bg2["start2"] // bin_size
    bg2["end2"] = bg2["end2"] // bin_size

    # Get number of bins per chromosome
    fragsA = bg2[["chr1", "end1"]].rename(
        columns={"chr1": "chr", "end1": "end"}
    )
    fragsB = bg2[["chr2", "end2"]].rename(
        columns={"chr2": "chr", "end2": "end"}
    )
    frags = pd.concat([fragsA, fragsB])
    chroms = frags.groupby("chr", sort=False).apply(lambda x: max(x.end))

    # Shift by one to get starting bin, first one is zero
    chrom_start = chroms.shift(1)
    chrom_start[0] = 0
    chrom_start = pd.DataFrame(chrom_start.cumsum(), columns=["cumsum"])

    # Get frags indices
    bg2 = bg2.merge(chrom_start, left_on="chr1", right_index=True)
    bg2["frag1"] = bg2["start1"] + bg2["cumsum"]
    bg2 = bg2.merge(chrom_start, left_on="chr2", right_index=True)
    bg2["frag2"] = bg2["start1"] + bg2["cumsum_y"]

    # Build sparse matrix from fragment indices
    n = int(max(max(bg2["frag1"]), max(bg2["frag2"]))) + 1
    mat = coo_matrix(
        (bg2.contacts, (bg2.frag1, bg2.frag2)), shape=(n, n), dtype=np.float64
    )

    # Making full symmetric matrix if not symmetric already (e.g. upper triangle)
    r = mat.tolil()
    if (abs(r - r.T) > 1e-10).nnz != 0:
        r += r.T
        r.setdiag(r.diagonal() / 2)
    r = r.tocoo()
    r.eliminate_zeros()

    # Get chroms into a 1D array of bin starts
    chrom_start = np.array(chrom_start["cumsum"])

    print("sparse matrix loaded. Sent to pattern_detector.")
    return mat, chrom_start


def dense2sparse(M, format='coo'):
    format_dict = {'coo': lambda x: x,
                   'csr': csr_matrix,
                   'csc': csc_matrix,
                   'lil': lil_matrix}
    N = np.triu(M)
    shape = N.shape
    nonzeros = N.nonzero()
    rows, cols = nonzeros
    data = M[nonzeros]
    S = coo_matrix((data, (rows, cols)), shape=shape)
    matrix_format = format_dict[format]
    return matrix_format(S)
