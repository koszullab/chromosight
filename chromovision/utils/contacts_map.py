# Implementation of a contact map class.
from __future__ import absolute_import
from . import io as cio
from . import preprocessing as preproc
import os
import sys
import numpy as np
import pandas as pd


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
    good_bins : list of arrays of ints
        List of two arrays containing indices of detectable rows and columns.
    chroms : pandas.DataFrame
        Table containing chromosome related informations.
    bins : pandas.DataFrame
        Table containing bin related informations.
    resolution : int
        The basepair resolution of the Hi-C matrix.
    inter : bool
        Whether interchromosomal matrices should be stored.
    """

    def __init___(self, path, inter=False):
        # Load Hi-C matrix and associated metadata
        self.matrix, self.chroms, self.bins, self.resolution = self.load_data(path)

        self.inter = inter

        # Preprocess the full genome matrix
        self.detectable_bins = preproc.get_detectable_bins(self.matrix)
        self.matrix = preproc.normalize(self.matrix, good_bins=self.good_bins)

        # Create sub matrices objects

    def load_data(self, mat_path):
        """Load contact, bin and chromosome informations from input path"""
        # Define functions to use for each format
        format_loader = {"bg2": cio.load_bedgraph2d, "cool": cio.load_cool}
        # Guess file format fron file name
        extension = os.path.splitext(mat_path)[-1].lstrip(".")
        print("loading: ", mat_path)

        # Load contact map and chromosome start bins coords
        try:
            sub_mat_df, chrom_start, bins, resolution = format_loader[extension](
                mat_path
            )
        except KeyError as e:
            sys.stderr.write(
                f"Unknown format: {extension}. Must be one of {format_loader.keys()}\n"
            )
            raise e
        return sub_mat_df, chrom_start, bins, resolution

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
                np.zeros((int(n_chroms ** 2 / 2 + n_chroms / 2), 3), dtype=str),
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
                if i1 >= i2:
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
                        )
                        sub_mat_idx += 1
                    elif self.inter:
                        sub_mats.contact_map[sub_mat_idx] = ContactMap(
                            sub_mat,
                            resolution=self.resolution,
                            detectable_bins=sub_mat_detectable_bins,
                            inter=True,
                        )
                        sub_mat_idx += 1
        return sub_mats

    def get_full_mat_pattern(self, c1, c2, patterns):
        """
        Converts bin indices of a list of patterns from an submatrix into their
        value in the original full-genome matrix.

        Parameters
        ----------
        c1 : str
            Name of the first chromosome of the sub matrix (rows).
        c2 : str
            Name of the second chromosome of the sub matrix (cols).
        pattern : numpy.array
            A 2D array of pattern coordinates. Each row is a pattern, column 0
            is the bin on c1 and column 1 is the bin on c2.
        """

        # Get start bin for chromosomes of interest
        startA = self.chroms.loc[self.chroms.name == c1, "start"]
        startB = self.chroms.loc[self.chroms.name == c2, "start"]
        # Shift index by start bin of chromosomes
        patterns[:, 0] += startA
        patterns[:, 1] += startB
        return patterns


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
    
    """

    def __init__(self, matrix, resolution, detectable_bins=None, inter=False):
        # If detectable were not provided, compute them from the input matrix
        if detectable_bins is None:
            detectable_bins = preproc.get_detectable_bins(matrix)
        self.detectable_bins = detectable_bins
        self.matrix = matrix
        self.resolution = resolution
        self.inter = inter

        # Apply preprocessing steps on the input matrix
        if self.inter:
            self.matrix = self.preprocess_inter_matrix(self.matrix)
        else:
            self.matrix = self.preprocess_intra_matrix(
                self.matrix, self.detectable_bins
            )

    def preprocess_inter_matrix(self, sub_mat):
        return preproc.ztransform(sub_mat)

    def preprocess_intra_matrix(self, sub_mat, sub_mat_detectable_bins):
        # Remove speckles (outlier pixels)
        sub_mat = preproc.despeckle(sub_mat, th2=3)
        # Compute signal to noise ratio at all diagonals
        snr_dist = preproc.signal_to_noise_threshold(sub_mat, sub_mat_detectable_bins)
        # Define max_dist based on snr and pattern config
        if self.max_dist is None:
            sub_mat_max_dist = snr_dist
        else:
            sub_mat_max_dist = min(self.max_dist, snr_dist)
        # Detrend matrix for power law
        sub_mat = preproc.detrend(sub_mat, sub_mat_detectable_bins)
        # Remove pixels further than max_dist
        for diag in range(sub_mat_max_dist + 1, sub_mat.shape[0]):
            sub_mat.setdiag(0, diag)

        return sub_mat
