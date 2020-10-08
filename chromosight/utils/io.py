#!/usr/bin/env python3
# coding: utf-8
"""Chromosight's io submodule contains input/output related functions to load
contact matrices in cool format, and save output patterns coordinates and
windows."""

import pandas as pd
import numpy as np
import pathlib
import sys
import csv
import shutil
from urllib.request import urlopen
import json
from jsonschema import validate, ValidationError
from os.path import join, dirname, isdir
from scipy.sparse import coo_matrix, triu


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

    .. code-block:: JSON

    {
        "name": str,
        "kernels": [
            str,
            ...
        ],
        "max_dist": int,
        "min_dist": int,
        "max_iterations": int,
        "max_perc_zero": float,
        "max_perc_undetected": float,
        "pearson": float
        "resolution": int
    }


    The kernels field should contain a list of path to kernel matrices to be
    loaded. These path should be relative to the config file. When loaded, the
    kernel field will contain the target matrices as 2D numpy arrays.

    The kernel matrices files themselves are raw tsv files containing a dense
    matrix of numeric value as read by the numpy.loadtxt function.

    Other fields are:

    * name : Name of the pattern
    * max_dist : maximum distance in basepairs to scan from the diagonal
    * max_iterations: maximum number of scanning iterations to perform
    * max_perc_zero: Maximum percentage of empty (0) pixels to include a pattern
    * max_perc_zero: Maximum percentage of missing (nan) pixels to include a pattern
    * pearson: Increasing this value reduces false positive patterns.
    * resolution: Basepair resolution for the kernel matrix.

    Parameters
    ----------
    kernel : str
        The name of the built-in pattern configuration to load if custom is
        False. Otherwise, the path to the custom JSON configuration file to
        load.
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
            "max_perc_undetected": {"type": "number", "minimum": 0},
            "max_perc_zero": {"type": "number", "minimum": 0},
            "pearson": {"type": "number"},
            "resolution": {"type": "number"},
        },
        "required": [
            "name",
            "kernels",
            "min_dist",
            "max_dist",
            "max_iterations",
            "min_separation",
            "pearson",
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


def write_patterns(coords, output_prefix, dec=10):
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
        Number of decimals to keep in correlation scores and p-values.
    """
    file_path = output_prefix + ".tsv"
    coords.to_csv(file_path, sep="\t", index=None, float_format=f"%.{dec}f")


def save_windows(windows, output_prefix, fmt="json"):
    """
    Write windows surrounding detected patterns to a npy or json file.  The
    file contains a 3D array where windows are piled on axis 0, matrix rows are
    on axis 1 and columns on axis 2.

    Parameters
    ----------
    windows : numpy.array of floats
        3D numpy array with axes (windows, rows, columns).
    output_prefix : str
        Output path where the file will be saved, an extension will be added
        based on the value of "format".
    format : str
        Format in which to save windows. Can be either npy for
        numpy's binary format, or json for a general purpose text
        format.
    """
    if fmt == "npy":
        file_path = output_prefix + ".npy"
        np.save(file_path, windows)
    elif fmt == "json":
        file_path = output_prefix + ".json"
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
        The content of the 2D BED file as a dataframe with 6 columns. Header
        will be: chrom1, start1, end1, chrom2, start2, end2.
    """
    header_finder = csv.Sniffer()
    with open(path) as f:
        header = header_finder.has_header(f.read(65536))
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
    bed2d.chrom1 = bed2d.chrom1.astype(str)
    bed2d.chrom2 = bed2d.chrom2.astype(str)
    inverted = (bed2d.start2 < bed2d.start1) & (bed2d.chrom1 == bed2d.chrom2)
    bed2d.start1[inverted], bed2d.start2[inverted] = (
        bed2d.start2[inverted],
        bed2d.start1[inverted],
    )
    bed2d.end1[inverted], bed2d.end2[inverted] = (
        bed2d.end2[inverted],
        bed2d.end1[inverted],
    )

    return bed2d


def download_file(url, file, length=16 * 1024):
    req = urlopen(url)
    with open(file, "wb") as fp:
        shutil.copyfileobj(req, fp, length)


def check_prefix_dir(prefix):
    """Checks for existence of the parent directory of an output prefix"""
    out_dir = dirname(prefix)
    if out_dir and not isdir(out_dir):
        raise OSError(f"Directory {out_dir} does not exist.")
