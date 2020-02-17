# Implementation of a contact map class.
from __future__ import absolute_import
from . import io as cio
from . import preprocessing as preproc
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DumpMatrix:
    """
    This class is used as a decorator to wrap methods and generate dump files
    of their class' "matrix" atribute. The matrix should be a scipy.sparse
    matrix and will be saved in npy format. If the instance has a "name" 
    attribute, the full dump path will be:
        self.dump / self.name_basename.npy
    if the instance has no name attribute, the dump path will be:
        self.dump / basename.npy
    If the instance has no matrix or dump attribute, no action is performed.

    Parameters
    ----------
    dump_name : str, os.PathLike object or None
        The basename of the file where to save the dump. If None, no action is
        performed.
    """

    def __init__(self, dump_name):
        """Executed only at method definition when used as method wrapper"""
        self.dump_name = dump_name

    def __call__(self, fn, *args, **kwargs):
        def decorated_fn(*args, **kwargs):
            """
            Executed at run time of the wrapped method.
            Executes the input function with its arguments, then dumps the
            matrix to target path. Note args[0] will always denote the instance
            of the wrapped method.
            """
            # Execute wrap function and store result
            res = fn(*args, **kwargs)
            # Instance of the wrapped method
            inst = args[0]
            # Define dump path based on instance's attributes
            if (
                hasattr(inst, "matrix")
                and inst.dump is not None
                and self.dump_name is not None
            ):
                if hasattr(inst, "name"):
                    dump_path = (
                        Path(inst.dump) / f"{inst.name}_{self.dump_name}"
                    )
                else:
                    dump_path = Path(inst.dump) / f"{self.dump_name}"
                print(
                    f"Dumping matrix after executing {fn.__name__} to {dump_path}"
                )
                # Save updated matrix to dump path
                sp.save_npz(dump_path, inst.matrix)
            return res

        return decorated_fn


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
    dump : str
        Base path where dump files will be generated. None means no dump.
    smooth : bool
        Whether isotonic regression should be used to smooth the signal for
        detrending intrachromosomal sub matrices. This  will reduce noise at
        long ranges but assumes contacts can only decrease with distance from
        the diagonal. Do not use with circular chromosomes.
    """

    def __init__(
        self, path, inter=False, kernel_config=None, dump=None, smooth=False
    ):
        # Load Hi-C matrix and associated metadata
        try:
            self.dump = Path(dump)
            os.makedirs(self.dump, exist_ok=True)
        except TypeError:
            self.dump = None
        self.matrix, self.chroms, self.bins, self.resolution = self.load_data(
            path
        )
        self.smooth = smooth
        self.kernel_config = kernel_config
        self.sub_mats = None
        self.detectable_bins = np.array(range(self.matrix.shape[0]))
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
        except (ValueError, TypeError):

            self.max_dist = None
            self.largest_kernel = 3

    @DumpMatrix("02_normalized")
    def normalize(self, n_mads=5, iterations=10):
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
            self.matrix, good_bins=self.detectable_bins, iterations=iterations
        )
        print("Whole genome matrix normalized")

    @DumpMatrix("01_subsampled")
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
                    raise ValueError(
                        "Error: Subsample must be strictly positive."
                    )
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
                    "Error: Subsample must be a number of reads or proportion."
                )
                raise

    def load_data(self, mat_path):
        """Load contact, bin and chromosome informations from input path"""
        # Define functions to use for each format
        format_loader = {
            "bg2": cio.load_bedgraph2d,
            "cool": cio.load_cool,
            "iobg2": cio.load_bedgraph2d,
        }
        # Guess file format fron file name
        try:
            extension = os.path.splitext(mat_path)[-1].lstrip(".")
        except TypeError:
            # if mat_path isn't an actual file path but a file object
            extension = "iobg2"
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
        if resolution is None:
            raise ValueError(
                "Input matrices must have a fixed genomic bin size."
            )
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
        sys.stderr.write("Preprocessing sub-matrices...\n")
        sub_mat_idx = 0
        for i1, r1 in self.chroms.iterrows():
            for i2, r2 in self.chroms.iterrows():
                s1, e1 = r1.start_bin, r1.end_bin
                s2, e2 = r2.start_bin, r2.end_bin
                if i1 == i2 or (i1 < i2 and self.inter):
                    cio.progress(
                        sub_mat_idx,
                        sub_mats.shape[0],
                        f"{r1['name']}-{r2['name']}",
                    )
                    # Subset intra / inter sub_matrix and matching detectable bins
                    sub_mat = matrix[s1:e1, s2:e2]
                    sub_mat_detectable_bins = (
                        d[(d >= s1) & (d < e1)] - s1,
                        d[(d >= s2) & (d < e2)] - s2,
                    )
                    if i1 == i2:
                        sub_mats.contact_map[sub_mat_idx] = ContactMap(
                            sub_mat,
                            detectable_bins=sub_mat_detectable_bins,
                            inter=False,
                            max_dist=self.max_dist,
                            largest_kernel=self.largest_kernel,
                            dump=self.dump,
                            name=f"{r1['name']}-{r2['name']}",
                            smooth=self.smooth,
                        )
                    else:
                        sub_mats.contact_map[sub_mat_idx] = ContactMap(
                            sub_mat,
                            detectable_bins=sub_mat_detectable_bins,
                            inter=True,
                            dump=self.dump,
                            name=f"{r1['name']}-{r2['name']}",
                        )
                    sub_mats.chr1[sub_mat_idx] = r1["name"]
                    sub_mats.chr2[sub_mat_idx] = r2["name"]
                    sub_mat_idx += 1
        cio.progress(
            sub_mat_idx,
            sub_mats.shape[0],
            f"{sub_mats.loc[sub_mat_idx-1, 'chr1']}-{sub_mats.loc[sub_mat_idx-1, 'chr2']}\n",
        )
        self.sub_mats = sub_mats
        print("Sub matrices extracted")

    def gather_sub_matrices(self):
        """Gathers all processed sub_matrices into a whole genome matrix """
        gathered = sp.csr_matrix(self.matrix.shape)
        # Define shortcut to extract bins for each chromosome
        c = self.chroms.set_index('name')
        chr_extent = lambda n: c.loc[n, ['start_bin', 'end_bin']].values
        # Fill empty whole genome matrix with processed submatrices
        for i1, r1 in self.sub_mats.iterrows():
            s1, e1 = chr_extent(r1.chr1)
            s2, e2 = chr_extent(r1.chr2)
            # Use each chromosome pair's sub matrix to fill the whole genome matrix
            gathered[s1:e1, s2:e2] = r1.contact_map.matrix
        gathered = sp.triu(gathered)
        return gathered

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
        
        Returns
        -------
        full_patterns : pandas.DataFrame
            A dataframe similar to the input, but with bins shifted to represent
            coordinates in the whole genome matrix.
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

    def get_sub_mat_pattern(self, chr1, chr2, patterns):
        """
        Converts bin indices of a list of patterns from the whole genome matrix
        into their value in the desired intra- or inter-chromosomal sub-matrix.

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
        
        Returns
        -------
        full_patterns : pandas.DataFrame
            A dataframe similar to the input, but with bins shifted to represent
            coordinates in the target sub-matrix.
        """
        sub_patterns = patterns.copy()
        # Get start bin for chromosomes of interest
        startA = self.chroms.loc[self.chroms.name == chr1, "start_bin"].values[
            0
        ]
        startB = self.chroms.loc[self.chroms.name == chr2, "start_bin"].values[
            0
        ]
        # Shift index by start bin of chromosomes
        sub_patterns.bin1 -= startA
        sub_patterns.bin2 -= startB
        return sub_patterns

    def bins_to_coords(self, bin_idx):
        """
        Converts a list of bin IDs to genomic coordinates based on the whole genome
        contact map.

        Parameters
        ----------
        bin_idx : numpy.array of ints
            A list of bin numbers corresponding to rows or columns of the whole
            genome matrix.

        Returns
        -------
        pandas.DataFrame :
            A subset of the bins dataframe, with columns chrom, start, end where
            chrom is the chromosome name (str), and start and end are the genomic
            coordinates of the bin (int).
        
        """
        return self.bins.iloc[bin_idx, :]

    def coords_to_bins(self, coords):
        """
        Converts genomic coordinates to a list of bin ids based on the whole genome
        contact map.

        Parameters
        ----------
        coords : pandas.DataFrame
            Table of genomic coordinates, with columns chrom, pos.

        Returns
        -------
        numpy.array of ints :
            Indices in the whole genome matrix contact map.
        
        """
        coords.pos = (coords.pos // self.resolution) * self.resolution
        # Coordinates are merged with bins, both indices are kept in memory so that
        # the indices of matching bins can be returned in the order of the input
        # coordinates
        idx = (
            self.bins.reset_index()
            .rename(columns={"index": "bin_idx"})
            .merge(
                coords.reset_index().rename(columns={"index": "coord_idx"}),
                left_on=["chrom", "start"],
                right_on=["chrom", "pos"],
                how='right',
            )
            .set_index("bin_idx")
            .sort_values("coord_idx")
            .index.values
        )
        return idx


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
    inter : bool
        True if the matrix represents contacts between two different,
        False otherwise.
    max_dist : int
        Maximum distance (in bins) at which contact values should be analysed. Only
        valid for intrachromosomal matrices.
    dump : str
        Base path where dump files will be generated. None means no dump.
    name : str
        Name of the submatrix (used for dumping).
    smooth : bool
        Whether isotonic regression should be used to smooth the signal for
        detrending. This  will reduce noise at long ranges but assumes contacts
        can only decrease with distance from the diagonal. Do not use with
        circular chromosomes.
    """

    def __init__(
        self,
        matrix,
        name="",
        detectable_bins=None,
        inter=False,
        max_dist=None,
        largest_kernel=0,
        dump=None,
        smooth=False,
    ):
        self.smooth = smooth
        self.despeckle = False
        self.snr = False
        self.matrix = matrix
        self.inter = inter
        self.max_dist = max_dist
        self.name = name
        self.largest_kernel = largest_kernel
        self.dump = dump
        # If detectable were not provided, compute them from the input matrix
        if detectable_bins is None:
            detectable_bins = preproc.get_detectable_bins(
                self.matrix, inter=self.inter
            )
        self.detectable_bins = detectable_bins

        # Apply preprocessing steps on the input matrix
        if self.inter:
            self.preprocess_inter_matrix()
        else:
            self.preprocess_intra_matrix()

    @DumpMatrix("01_process_inter")
    def preprocess_inter_matrix(self):
        self.matrix /= np.median(self.matrix.data)

    def preprocess_intra_matrix(self):
        self.detrend()
        self.remove_diags()

    @DumpMatrix("01_detrended")
    def detrend(self):
        # Detrend matrix for power law
        self.matrix = preproc.detrend(
            self.matrix,
            max_dist=self.keep_distance,
            smooth=self.smooth,
            detectable_bins=self.detectable_bins[0],
        )

    @DumpMatrix("02_remove_diags")
    def remove_diags(self):
        # Create a new matrix from the diagonals below max dist (faster than removing them)
        self.matrix = preproc.diag_trim(
            self.matrix.todia(), self.keep_distance
        )
        self.matrix = self.matrix.tocoo()
        # Fill diagonals of the lower triangle that might overlap the kernel
        for i in range(1, min(self.matrix.shape[0], self.largest_kernel)):
            self.matrix.setdiag(1, -i)

    @property
    def keep_distance(self):
        # Define max scanning dist based on pattern config
        if self.max_dist is None:
            mat_max_dist = self.matrix.shape[0]
        else:
            mat_max_dist = min(self.max_dist, self.matrix.shape[0])
        # If we scan until a given distance, data values in a margin must be kept as well
        return mat_max_dist + (self.largest_kernel)
