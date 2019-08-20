#!/usr/bin/env python3
# coding: utf-8
"""Input/Output related functions
Load and save contact matrices in sparse format
"""
import pandas as pd
import numpy as np
import pathlib
import sys
import json
from os.path import join
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
    # Throw error if bins are not equally sized (e.g. restriction fragments)
    if bin_size != bg2.end1[0] - bg2.start[0]:
        sys.stderr.write("Error: Bins are not of equal size.")
        sys.exit(1)

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


def load_kernel_config(kernel, custom=False):
    """
    Load a kernel configuration from input JSON file.

    All parameters associated with the kernel along its kernel matrices are
    loaded into a dictionary.

    A kernel config file is a JSON file with the following structure:

    ```json
    {
        "name": str,
        "kernels": [
            str,
            ...
        ],
        "max_dist": int,
        "max_iterations": int,
        "max_perc_undetected": float,
        "precision": float
        "resolution": int
    }
    ```
    The kernels field should contain a list of path to kernel matrices to be
    loaded. These path should be relative to the config file. When loaded, the
    kernel field will contain the target matrices as 2D numpy arrays.

    The kernel matrices files themselves are raw tsv files containing a dense
    matrix of numeric value as read by the numpy.loadtxt function.

    Other fields are:
    * name : Name of the pattern
    * max_dist : maximum distance in basepairs to scan from the diagonal
    * max_iterations: maximum number of scanning iterations to perform
    * max_perc_undetected: Maximum percentage of undetected bins to include a pattern
    * precision: Increasing this value reduces false positive patterns.
    * resolution: Basepair resolution for the kernel matrix.

    Parameters
    ----------
    kernel : str
        The name of the built-in pattern configuration to load if custom is
        False. Otherwise, the path to the custom JSON configuration file to load.
    custom : bool
        Determines if a custom JSON configuration file must be loaded, or if a
        preset configuration is used.
    Returns
    -------
    pattern_kernels : list
        A list of array_likes corresponding to the loaded patterns.
    """

    # Custom kernel: use litteral path as config path
    if custom:
        config_path = kernel
    # Preset kernel: Find preset config file matching pattern name
    else:
        # Find chromovision installation directory and get kernel config path
        chromo_dir = pathlib.Path(__file__).parents[2]
        preset_kernel_dir = pathlib.Path(join(chromo_dir, "kernels"))
        # Preset config filename should be {pattern}.json
        config_path = join(preset_kernel_dir, f"{kernel}.json")

    # Try parsing config file
    try:
        with open(config_path, "r") as config:
            kernel_config = json.load(config)
    except FileNotFoundError as e:
        if custom:
            sys.stderr.write(
                f"Error: Kernel configuration file {config_path} does not exist.\n"
            )
        else:
            err_msg = f"Error: No preset configuration for pattern {kernel}.\n"
        sys.stderr.write(err_msg)
        raise e

    # Load kernel matrices using path in kernel config
    kernel_matrices = [None] * len(kernel_config["kernels"])
    # Keep track of the largest kernel
    largest_kernel = 0
    for i, kernel_path in enumerate(kernel_config["kernels"]):
        kernel_path = join(pathlib.Path(config_path).parent, kernel_path)
        kernel_matrices[i] = np.loadtxt(kernel_path)
        if kernel_matrices[i].shape[0] > largest_kernel:
            largest_kernel = kernel_matrices[i].shape[0]
    # Replace matrices path by their content in the config dictionary
    kernel_config["kernels"] = kernel_matrices

    # Set maximum distance parameter to bins using resolution
    # Make sure max distance is not smaller than kernel
    kernel_config["max_dist"] = max(
        kernel_config["max_dist"] // kernel_config["resolution"], largest_kernel
    )

    return kernel_config


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


def write_results(patterns_to_plot, pattern_name, output):
    file_name = pattern_name + ".txt"
    file_path = output / file_name
    with file_path.open("w") as outf:
        for tup in sorted([tup for tup in patterns_to_plot if "NA" not in tup]):
            outf.write(" ".join(map(str, tup)) + "\n")
