# Implementation of a contact map class.
from __future__ import absolute_import
from . import io as cio
from . import preprocessing as preproc
import os
import re
from time import time
import sys
import numpy as np
import pandas as pd
from scipy.sparse import dia_matrix


class HicGenome:
    """
    Class used to manage relationships between whole genome and intra- or inter-
    chromosomal Hi-C sub matrices. Also handles reading and writing data.

    Attributes:
    -----------
    matrix : scipy.sparse.coo_matrix
        The whole genome Hi-C contact map, in sparse format
    sub_mats : pandas.DataFrame
        Table containing intra- and optionally inter-chromosomal matrices.
    detectable_bins : list of arrays of ints
        List of two arrays containing indices of detectable rows and columns.
    chroms : pandas.DataFrame
        Table containing chromosome related informations.
    bins : pandas.DataFrame
        Table containing bin related informations.
    resolution : int
        The basepair resolution of the Hi-C matrix.
    inter : bool
        Whether interchromosomal matrices should be stored.
    kernel_config : dict
        Kernel configuration associated with the Hi-C genome
    max_dist : int
        Maximum scanning distance for convolution during pattern detection.

    """

    def __init__(self, path, inter=False, kernel_config=None):
        # Load Hi-C matrix and associated metadata
        self.matrix, self.chroms, self.bins, self.resolution = self.load_data(
            path
        )
        self.kernel_config = kernel_config
        self.sub_mats = None
        self.detectable_bins = None
        self.inter = inter
        self.compute_max_dist()

    def compute_max_dist(self):
        """Use the kernel config to compute max_dist"""
        # Convert maximum scanning distance to bins using resolution
        try:
            self.max_dist = max(
                self.kernel_config["max_dist"] // self.resolution, 1
            )
            # Get the size of the largest convolution kernel
            self.largest_kernel = max(
                [s.shape[0] for s in self.kernel_config["kernels"]]
            )
        except ValueError:

            self.max_dist = None
            self.largest_kernel = 3

    def normalize(self, n_mads=5):
        """
        Finds detectable bins and applies ICE normalisation to the whole genome
        matrix.

        Parameters
        ----------
        n_mads : float
            Maximum number of median absoluted deviations (MADs) below the
            median of the distribution of logged bin sums to consider a bin
            detectable.
        """
        # Preprocess the full genome matrix
        self.detectable_bins = preproc.get_detectable_bins(
            self.matrix, n_mads=n_mads
        )[0]
        print(
            f"Found {len(self.detectable_bins)} / {self.matrix.shape[0]} detectable bins"
        )
        self.matrix = preproc.normalize(
            self.matrix, good_bins=self.detectable_bins
        )
        print("Whole genome matrix normalized")

    def subsample(self, sub):
        """
        Subsample contacts from the Hi-C matrix.
        
        Parameters
        ----------
        sub : float
            Proportion of contacts to subsample from the matrix if between 0 and 1.
            Number of contacts to keep if above 1. Keep all contacts if None.
        """
        if sub is not None:
            try:
                subsample = float(sub)
                if subsample < 0:
                    sys.stderr.write(
                        "Error: Subsample must be strictly positive.\n"
                    )
                    sys.exit(1)
                if subsample < 1:
                    subsample *= self.matrix.sum()
                if subsample < self.matrix.sum():
                    subsample = int(subsample)
                    print(f"Subsampling {subsample} contacts from matrix")
                    self.matrix = preproc.subsample_contacts(
                        self.matrix, int(subsample)
                    )
                else:
                    print(
                        "Skipping subsampling: Value is higher than the number of contacts in the matrix."
                    )
            except ValueError:
                sys.stderr.write(
                    "Error: Subsample must be a number of reads or proportion.\n"
                )
                sys.exit(1)

    def load_data(self, mat_path):
        """Load contact, bin and chromosome informations from input path"""
        # Define functions to use for each format
        format_loader = {"bg2": cio.load_bedgraph2d, "cool": cio.load_cool}
        # Guess file format fron file name
        extension = os.path.splitext(mat_path)[-1].lstrip(".")
        if not len(extension) and re.search(r"mcool::", mat_path):
            extension = "cool"
        print("loading: ", mat_path)

        # Load contact map and chromosome start bins coords
        try:
            sub_mat_df, chroms, bins, resolution = format_loader[extension](
                mat_path
            )
        except KeyError as e:
            sys.stderr.write(
                f"Unknown format: {extension}. Must be one of {format_loader.keys()}\n"
            )
            raise e
        return sub_mat_df, chroms, bins, resolution

    def make_sub_matrices(self):
        """
        Generates a table of Hi-C sub matrices. Each sub matrix is either intra
        or interchromosomal. The table has 3 columns: chr1, chr2 and contact_map.
        The contact_map column contains instances of the ContactMap class.

        Returns
        -------
        pandas.DataFrame:
            The table of sub matrices which will contain n_chrom rows if the inter
            attribute is set to false, or (n_chrom^2) / 2 + n_chroms / 2 if inter
            is True (that is, the upper triangle matrix).
        """
        # Convert whole genome matrix to CSR for indexing
        matrix = self.matrix.tocsr()
        # Create an empty dataframe to store sub matrix info
        sub_cols = ["chr1", "chr2", "contact_map"]
        n_chroms = self.chroms.shape[0]
        if self.inter:
            sub_mats = pd.DataFrame(
                np.zeros(
                    (int(n_chroms ** 2 / 2 + n_chroms / 2), 3), dtype=str
                ),
                columns=sub_cols,
            )
        else:
            sub_mats = pd.DataFrame(
                np.zeros((n_chroms, 3), dtype=str), columns=sub_cols
            )

        d = self.detectable_bins

        # Loop over all possible combinations of chromosomes
        # in the upper triangle matrix
        sub_mat_idx = 0
        for i1, r1 in self.chroms.iterrows():
            for i2, r2 in self.chroms.iterrows():
                s1, e1 = r1.start_bin, r1.end_bin
                s2, e2 = r2.start_bin, r2.end_bin
                if i1 == i2 or (i1 < i2 and self.inter):
                    # Subset intra / inter sub_matrix and matching detectable bins
                    sub_mat = matrix[s1:e1, s2:e2]
                    sub_mat_detectable_bins = (
                        d[(d >= s1) & (d < e1)] - s1,
                        d[(d >= s2) & (d < e2)] - s2,
                    )
                    if i1 == i2:
                        sub_mats.contact_map[sub_mat_idx] = ContactMap(
                            sub_mat,
                            resolution=self.resolution,
                            detectable_bins=sub_mat_detectable_bins,
                            inter=False,
                            max_dist=self.max_dist,
                            largest_kernel=self.largest_kernel,
                        )
                    else:
                        sub_mats.contact_map[sub_mat_idx] = ContactMap(
                            sub_mat,
                            resolution=self.resolution,
                            detectable_bins=sub_mat_detectable_bins,
                            inter=True,
                        )
                    sub_mats.chr1[sub_mat_idx] = r1["name"]
                    sub_mats.chr2[sub_mat_idx] = r2["name"]
                    sub_mat_idx += 1
                    print(f"Processed {r1['name']}-{r2['name']} sub matrix.")
        self.sub_mats = sub_mats
        print("Sub matrices extracted")

    def get_full_mat_pattern(self, chr1, chr2, patterns):
        """
        Converts bin indices of a list of patterns from an submatrix into their
        value in the original full-genome matrix.

        Parameters
        ----------
        chr1 : str
            Name of the first chromosome of the sub matrix (rows).
        chr2 : str
            Name of the second chromosome of the sub matrix (cols).
        pattern : pandas.DataFrame
            A dataframme of pattern coordinates. Each row is a pattern and
            columns should be bin1 and bin2, for row and column coordinates in
            the Hi-C matrix, respectively.
        """
        full_patterns = patterns.copy()
        # Get start bin for chromosomes of interest
        startA = self.chroms.loc[self.chroms.name == chr1, "start_bin"].values[
            0
        ]
        startB = self.chroms.loc[self.chroms.name == chr2, "start_bin"].values[
            0
        ]
        # Shift index by start bin of chromosomes
        full_patterns.bin1 += startA
        full_patterns.bin2 += startB
        return full_patterns

    def bin_to_coords(self, bin_idx):
        """
        Converts a bin ID to a genomic coordinates based on the whole genome
        contact map.

        Parameters
        ----------
        bin_idx : int
            A bin number corresponding to a row or column of the whole genome matrix.

        Returns
        -------
        tuple :
            A tuple of the form (chrom, start, end) where chrom is the chromosome
            name (str), and start and end are the genomic coordinates of the bin (int).
        
        """
        return self.bins.iloc[bin_idx, :]


class ContactMap:
    """
    Class to store and manipulate a simple Hi-C matrix, either intra or
    inter-chromosomal.

    Attributes:
    -----------
    matrix : scipy.sparse.coo_matrix
        Sparse matrix object containing the Hi-C contact data. Either intra or
        interchromosomal matrix.
    detectable_bins : tuple of arrays
        List containing two arrays (rows and columns) of indices from bins
        considered detectable in the matrix.
    resolution : int
        The basepair resolution of the matrix
    inter : bool
        True if the matrix represents contacts between two different,
        False otherwise.
    max_dist : int
        Maximum distance (in bins) at which contact values should be analysed. Only
        valid for intrachromosomal matrices.
    
    """

    def __init__(
        self,
        matrix,
        resolution,
        detectable_bins=None,
        inter=False,
        max_dist=None,
        largest_kernel=0,
    ):
        self.despeckle = False
        self.snr = False
        self.matrix = matrix
        self.resolution = resolution
        self.inter = inter
        self.max_dist = max_dist
        self.largest_kernel = largest_kernel
        # If detectable were not provided, compute them from the input matrix
        if detectable_bins is None:
            detectable_bins = preproc.get_detectable_bins(
                self.matrix, inter=self.inter
            )
        self.detectable_bins = detectable_bins

        # Apply preprocessing steps on the input matrix
        if self.inter:
            self.matrix = self.preprocess_inter_matrix()
        else:
            self.matrix = self.preprocess_intra_matrix()

    def preprocess_inter_matrix(self):
        return preproc.ztransform(self.matrix)

    def preprocess_intra_matrix(self):
        # Remove speckles (outlier pixels)
        if self.despeckle:
            sub_mat = preproc.despeckle(self.matrix, th2=3)
        else:
            sub_mat = self.matrix
        if self.snr:
            # Compute signal to noise ratio at all diagonals
            snr_dist = preproc.signal_to_noise_threshold(
                sub_mat, detectable_bins=self.detectable_bins[0]
            )
        else:
            snr_dist = sub_mat.shape[0]
        # Define max scanning dist based on snr and pattern config
        if self.max_dist is None:
            sub_mat_max_dist = snr_dist
        else:
            sub_mat_max_dist = min(self.max_dist, snr_dist)
        # If we scan until a given distance, data values in a margin must be kept as well
        keep_distance = sub_mat_max_dist + (self.largest_kernel)
        # Detrend matrix for power law
        sub_mat = preproc.detrend(
            sub_mat,
            max_dist=self.max_dist,
            detectable_bins=self.detectable_bins[0],
        )

        # Create a new matrix from the diagonals below max dist (faster than removing them)
        sub_mat = preproc.diag_trim(sub_mat.todia(), keep_distance)
        sub_mat = sub_mat.tocoo()
        # Fill diagonals of the lower triangle that might overlap the kernel
        for i in range(1, min(sub_mat.shape[0], self.largest_kernel)):
            sub_mat.setdiag(sub_mat.diagonal(i), -i)

        return sub_mat
