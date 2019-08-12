#!/usr/bin/env python3
# coding: utf-8
"""Input/Output related functions
Load and save contact matrices in sparse format
"""
import pandas as pd
import numpy as np
import pathlib
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix, csr_matrix, triu


def load_bedgraph2d(mat_path):
    """
    Loads a sparse Hi-C contact matrix in 2D bedgraph format into a sparse
    matrix and an array of chromosome start bins. Expects the file to have 7
    space-separated columns: chr1, start1, end1, chr2, start2, end2, contacts.
    Expects the file to contain same-size fragments (and not restriction
    fragments)

    Parameters
    ----------
    mat_path : str
        Path to the matrix in 2D bedgraph format.

    Returns
    -------
    mat: scipy.sparse.coo_matrix
        Output sparse matrix in coordinate format
    chrom_start : numpy.array
        1D array of starting bins for each chromosome.
    """
    bg2 = pd.read_csv(mat_path, delimiter="\t", header=None)
    bg2.head()
    bg2.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2", "contacts"]

    # estimate bin size from file
    bin_size = np.median(bg2.end1 - bg2.start1).astype(int)

    # Get binID (within chromosome) from base pairs
    bg2["bin1"] = bg2["start1"] // bin_size
    bg2["bin2"] = bg2["start2"] // bin_size
    # Get number of bins per chromosome
    binsA = bg2[["chr1", "start1", "end1"]].rename(
        columns={"chr1": "chrom", "start1": "start", "end1": "end"}
    )
    binsB = bg2[["chr2", "start2", "end2"]].rename(
        columns={"chr2": "chrom", "start2": "start", "end2": "end"}
    )
    bins = pd.concat([binsA, binsB])
    # Get size of each chromosome in bins
    chromsizes = bins.groupby("chrom").apply(
        lambda x: np.int64(max(x.start // bin_size + 1))
    )

    # Rebuild bin table based on chromosome size and binsize info
    # so that potentially missing bins are included
    bins = pd.DataFrame(
        {
            "start": [
                start for chromsize in chromsizes.values for start in range(chromsize)
            ]
        }
    )
    bins["start"] *= bin_size
    bins["end"] = bins["start"] + bin_size
    bins["chrom"] = np.repeat(np.array(chromsizes.index), np.array(chromsizes.values))

    # Shift chromsizes by one to get starting bin, first one is zero
    chrom_start = chromsizes.shift(1)
    chrom_start[0] = 0
    # Make chromsize cumulative to get start bin of each chrom
    chrom_start = pd.DataFrame(chrom_start.cumsum(), columns=["cumsum"], dtype=np.int)

    # Get frags indices
    bg2 = bg2.merge(chrom_start, left_on="chr1", right_index=True)
    # Make bin IDs absolute (not per chromosome)
    bg2["bin1"] += bg2["cumsum"]
    bg2 = bg2.merge(chrom_start, left_on="chr2", right_index=True)
    bg2["bin2"] += bg2["cumsum_y"]

    # Build sparse matrix from fragment indices
    n = int(max(max(bg2["bin1"]), max(bg2["bin2"]))) + 1
    mat = coo_matrix(
        (bg2.contacts, (bg2.bin1, bg2.bin2)), shape=(n, n), dtype=np.float64
    )

    # Get chroms into a 1D array of bin starts
    chrom_start = np.array(chrom_start["cumsum"])
    # Only keep upper triangle
    mat = triu(mat)
    bins = bins[["chrom", "start", "end"]]
    return mat, chrom_start, bins, bin_size


def load_cool(cool_path):
    """
    Reads a cool file into memory and parses it into a COO sparse matrix
    and an array with the starting bin of each chromosome.
    
    Parameters
    ----------
    cool : str
        Path to the input .cool file.

    Returns
    -------
    mat : scipy coo_matrix
        Hi-C contact map in COO format.
    chrom_start : numpy.array
        1D array of starting bins for each chromosome.
    """
    try:
        import cooler
    except ImportError:
        print(
            "The cooler package is required to use cool files. Please install it first."
        )
        raise

    c = cooler.Cooler(cool_path)  # pylint: disable=undefined-variable
    mat = c.pixels()[:]
    # Number of fragments  (bins) per chromosome
    bins = c.bins()[:]
    n_bins = bins.groupby("chrom", sort=False).count().start[:-1]
    n_bins = n_bins.astype(np.int64)
    # Starting bin of each chromosome
    chrom_start = np.insert(np.array(n_bins), 0, 0)
    # Make a sparse (COO) matrix from the pixels table
    n = int(max(np.amax(mat.bin1_id), np.amax(mat.bin2_id))) + 1
    shape = (n, n)
    mat = coo_matrix(
        (mat["count"], (mat.bin1_id, mat.bin2_id)), shape=shape, dtype=np.float64
    )
    # Only keep upper triangle
    mat = triu(mat)
    bins = bins[["chrom", "start", "end"]]
    return mat, chrom_start, bins, c.binsize


def load_kernels(pattern):
    """Load pattern kernels

    Look for one or several kernel file (in CSV format).

    Parameters
    ----------
    pattern : str
        The pattern type. Must be one of 'borders', 'loops' or 'centromeres',
        but partial matching is allowed.

    Returns
    -------
    pattern_kernels : list
        A list of array_likes corresponding to the loaded patterns.
    """

    pattern_path = pathlib.Path(pattern)
    if pattern in PATTERN_DISPATCHER.keys():
        pattern_globbing = PRESET_KERNEL_PATH.glob("*{}*".format(pattern))
    elif pattern_path.is_dir():
        pattern_globbing = pattern_path.glob("*")
    else:
        # Not finished ?
        pattern_globbing = (pattern_path,)
    for kernel_file in pattern_globbing:
        yield np.loadtxt(kernel_file, dtype=np.float)


def load_dense_matrix():
    # TODO: add support for loading dense tsv matrices assuming single chrom
    ...
    # return mat, chrom_start, bins, binsize


def dense2sparse(M, format="coo"):
    format_dict = {
        "coo": lambda x: x,
        "csr": csr_matrix,
        "csc": csc_matrix,
        "lil": lil_matrix,
    }
    N = np.triu(M)
    shape = N.shape
    nonzeros = N.nonzero()
    rows, cols = nonzeros
    data = M[nonzeros]
    S = coo_matrix((data, (rows, cols)), shape=shape)
    matrix_format = format_dict[format]
    sparse_mat = matrix_format(S)
    return triu(sparse_mat)


def write_results(patterns_to_plot, output):
    for pattern in patterns_to_plot:
        file_name = pattern + ".txt"
        file_path = output / file_name
        with file_path.open("w") as outf:
            for tup in sorted(
                [tup for tup in patterns_to_plot[pattern] if "NA" not in tup]
            ):
                outf.write(" ".join(map(str, tup)) + "\n")
