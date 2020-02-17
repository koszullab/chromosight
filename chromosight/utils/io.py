#!/usr/bin/env python3
# coding: utf-8
"""Input/Output related functions
Load and save contact matrices in sparse format
"""
import pandas as pd
import numpy as np
import pathlib
import sys
import csv
import json
from jsonschema import validate, ValidationError
from os.path import join
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix, csr_matrix, triu
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


def load_bedgraph2d(mat_path):
    """
    Loads a sparse Hi-C contact matrix in 2D bedgraph format into a sparse
    matrix and an array of chromosome start bins. Expects the file to have 7
    space-separated columns: chr1, start1, end1, chr2, start2, end2, contacts.
    Expects the file to contain same-size fragments (and not restriction
    fragments)

    Parameters
    ----------
    mat_path : str, file-like object
        Path to the matrix in 2D bedgraph format.

    Returns
    -------
    mat: scipy.sparse.coo_matrix
        Output sparse matrix in coordinate format
    chroms : pandas.DataFrame
        Table of chromosome information. Each row contains the name, length,
        first and last bin of a chromosome.
    bins : pandas.DataFrame
        Table of genomic bins information. Each row contains the chromosome,
        genomic start and end coordinates of a matrix bin.
    bin_size : int
        Matrix resolution. Corresponds to the number of base pairs per matrix bin.
    """

    # Check if input is file-like object containing a csv (which has 8 columns
    # instead of 7 and whose 0th column needs to be trimmed)
    if hasattr(mat_path, "read"):
        bg2 = pd.read_csv(mat_path, header=None, usecols=list(range(1, 8)))
    else:
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

    # Throw error if bins are not equally sized (e.g. restriction fragments)
    if bin_size != bg2.end1[0] - bg2.start1[0]:
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
                start
                for chromsize in chromsizes.values
                for start in range(chromsize)
            ]
        }
    )
    bins["start"] *= bin_size
    bins["end"] = bins["start"] + bin_size
    bins["chrom"] = np.repeat(
        np.array(chromsizes.index), np.array(chromsizes.values)
    )

    # Shift chromsizes by one to get starting bin, first one is zero
    chrom_start = chromsizes.shift(1)
    chrom_start[0] = 0
    # Make chromsize cumulative to get start bin of each chrom
    chrom_start = pd.DataFrame(
        chrom_start.cumsum(), columns=["cumsum"], dtype=np.int
    )

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
    chrom_end = np.append(chrom_start[1:], chrom_start[-1] + chromsizes[-1])
    chroms = pd.DataFrame(
        {
            "name": chromsizes.index,
            "length": chromsizes.values,
            "start_bin": chrom_start,
            "end_bin": chrom_end,
        }
    )
    # Only keep upper triangle
    mat = triu(mat)
    bins = bins[["chrom", "start", "end"]]
    return mat, chroms, bins, bin_size


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
    mat: scipy.sparse.coo_matrix
        Output sparse matrix in coordinate format
    chroms : pandas.DataFrame
        Table of chromosome information. Each row contains the name, length,
        first and last bin of a chromosome.
    bins : pandas.DataFrame
        Table of genomic bins information. Each row contains the chromosome,
        genomic start and end coordinates of a matrix bin.
    bin_size : int
        Matrix resolution. Corresponds to the number of base pairs per matrix bin.
    """
    try:
        import cooler
    except ImportError:
        print(
            "The cooler package is required to use cool files. Please install it first."
        )
        raise

    c = cooler.Cooler(cool_path)  # pylint: disable=undefined-variable
    if c.binsize is None:
        raise ValueError("The cool file must have equally sized bins")
    mat = c.pixels()[:]
    # Number of fragments  (bins) per chromosome
    bins = c.bins()[:]
    chroms = c.chroms()[:]
    # Number of bins per chromosome
    n_bins = bins.groupby("chrom", sort=False).count().start
    n_bins = n_bins.astype(np.int64)
    # Starting bin of each chromosome
    chrom_start = np.insert(np.array(n_bins), 0, 0)
    # Make chromstart cumulative
    chrom_start = np.cumsum(chrom_start)
    # Make a sparse (COO) matrix from the pixels table
    n = int(max(np.amax(mat.bin1_id), np.amax(mat.bin2_id))) + 1
    shape = (n, n)
    mat = coo_matrix(
        (mat["count"], (mat.bin1_id, mat.bin2_id)),
        shape=shape,
        dtype=np.float64,
    )
    # Only keep upper triangle
    mat = triu(mat)
    bins = bins[["chrom", "start", "end"]]
    chroms["start_bin"] = chrom_start[:-1]
    chroms["end_bin"] = chrom_start[1:]
    return mat, chroms, bins, c.binsize


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
        "min_dist": int,
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
    kernel_config : dict
        A dictionary containing a key: value pair for each parameter as well as 
        list of kernel matrices under key 'kernels'.
    """
    # Define a schema for kernel configuration validation
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "kernels": {"type": "array", "items": {"type": "string"}},
            "min_dist": {"type": "number", "minimum": 0},
            "max_dist": {"type": "number", "minimum": 0},
            "max_iterations": {"type": "number", "minimum": 0},
            "min_separation": {"type": "number", "minimum": 1},
            "precision": {"type": "number"},
            "resolution": {"type": "number"},
        },
        "required": [
            "name",
            "kernels",
            "min_dist",
            "max_dist",
            "max_iterations",
            "min_separation",
            "precision",
            "resolution",
        ],
    }
    # Custom kernel: use litteral path as config path
    if custom:
        config_path = kernel
    # Preset kernel: Find preset config file matching pattern name
    else:
        # Find chromosight installation directory and get kernel config path
        chromo_dir = pathlib.Path(__file__).parents[1]
        preset_kernel_dir = pathlib.Path(join(chromo_dir, "kernels"))
        # Preset config filename should be {pattern}.json
        config_path = join(preset_kernel_dir, f"{kernel}.json")

    # Try parsing config file
    try:
        with open(config_path, "r") as config:
            kernel_config = json.load(config)
    except FileNotFoundError as e:
        if custom:
            err_msg = f"Error: Kernel configuration file {config_path} does not exist.\n"
        else:
            err_msg = f"Error: No preset configuration for pattern {kernel}.\n"
        sys.stderr.write(err_msg)
        raise e

    # Check that JSON file is valid
    try:
        validate(kernel_config, schema)
    except ValidationError:
        sys.stderr.write("Invalid kernel configuration\n")
        raise

    # Load kernel matrices using path in kernel config
    kernel_matrices = [None] * len(kernel_config["kernels"])
    for i, kernel_path in enumerate(kernel_config["kernels"]):
        kernel_path = join(pathlib.Path(config_path).parent, kernel_path)
        kernel_matrices[i] = np.loadtxt(kernel_path)
    # Replace matrices path by their content in the config dictionary
    kernel_config["kernels"] = kernel_matrices

    return kernel_config


def write_patterns(coords, pattern_name, output_dir, dec=5):
    """
    Writes coordinates to a text file.

    Parameters
    ----------
    coords : pandas.DataFrame
        Pandas dataframe containing the coordinates and score of
        one detected pattern per row.
    pattern_name : str
        Name of the pattern. Will be the basename of the output
        file.
    output_dir : str
        Output path where the file will be saved.
    dec : int
        Number of decimals to keep in correlation scores.
    """
    file_name = pattern_name + ".txt"
    file_path = join(output_dir, file_name)
    coords.score = np.round(coords.score, dec)
    coords.to_csv(file_path, sep="\t", index=None)


def save_windows(windows, pattern_name, output_dir=".", format="json"):
    """
    Write windows surrounding detected patterns to a npy file.
    The file contains a 3D array where windows are piled on
    axis 0, matrix rows are on axis 1 and columns on axis 2.

    Parameters
    ----------
    windows : numpy.array of floats
        3D numpy array with axes (windows, rows, columns).
    pattern_name : str
        Name of the pattern. Will be the basename of the output
        file.
    output_dir : str
        Output path where the file will be saved.
    format : str
        Format in which to save windows. Can be either npy for
        numpy's binary format, or json for a general purpose text
        format.
    """
    # TODO: Adapt for windows of different sizes
    if format == "npy":
        file_name = pattern_name + ".npy"
        file_path = join(output_dir, file_name)
        np.save(file_path, windows)
    elif format == "json":
        import json

        file_name = pattern_name + ".json"
        file_path = join(output_dir, file_name)
        json_wins = {idx: win.tolist() for idx, win in enumerate(windows)}
        with open(file_path, "w") as handle:
            json.dump(json_wins, handle, indent=4)
    else:
        raise ValueError("window format must be either npy or json.")


def progress(count, total, status=""):
    """
    Basic progress bar in terminal.
    
    Parameters
    ----------
    count : float
        Current task id.
    total : float
        Maximum task id.
    status : str
        Info to write on the side of the bar.
    """
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    
    # Clear current line, write status and erase everything to the end of
    # the line using ANSI code
    sys.stderr.write("\r [%s] %s%s %s\033[K" % (bar, percents, "%", status))
    sys.stderr.flush()


def load_bed2d(path):
    """
    Loads only the first 6 columns of a 2D BED file. Will sniff for header
    and generate a default header only if none is present.
    Compatible with output of chromosight detect.

    Parameters
    ----------
    path : str
        The path to the 2D BED file.

    Returns
    -------
    pandas.DataFrame :
        The content of the 2D BED file as a dataframe with 6 columns. Header will
        be: chrom1, start1, end1, chrom2, start2, end2.
    """
    header_finder = csv.Sniffer()
    with open(path) as f:
        header = header_finder.has_header(f.read(1024))
    if header:
        bed2d = pd.read_csv(path, sep="\t", header=0, usecols=range(6))
    else:
        bed2d = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
            usecols=range(6),
        )
    # Make sure start1 is always the left anchor for intrachromosomal pairs
    inverted = (bed2d.start2 < bed2d.start1) & (bed2d.chrom1 == bed2d.chrom2)
    bed2d.start1[inverted], bed2d.start2[inverted] = (
        bed2d.start2[inverted],
        bed2d.start1[inverted]
    )

    return bed2d
