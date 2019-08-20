# Implementation of a contact map class.
from __future__ import absolute_import
from . import io as cio
from . import preprocessing as preproc
import os
import sys
import numpy as np


class ContactMap:

    """
    Stores all aspects of the contact map. Including a list of sub matrices
    (sub_mats) for intra- and inter- chromosomal contacts.

    E.g. for a matrix of 3 chromosomes,
    sub_mats would contain 6 sub matrices (3^2 + 3) / 2, for the 3 inter
    matrices in the upper triangle and the 3 intra matrices on the diagonal.
    they would be stored in sub_mats as follows:

    Given the full matrix of 3 chromosomes where tile ij means the subset of
    the matrix between chromosomes i and j, and XX means not stored:
    00 01 02
    XX 11 12
    XX XX 22

    sub_mats contains the following sub matrices: [ 00 01 02 11 12 22 ]
    in the case where interchrom attribute is set to False,
    sub_mats contains only: [00 11 22]

    Attributes
    ----------
    mat : scipy.sparse.coo_matrix
        The full contact map in sparse, upper triangle format
    chroms : numpy.array of ints
        A 2D numpy array containing boundaries of chromosomes. Number
        of rows is # chromosomes and there are two columns for the start
        and end of each chromosome.
    chrom_labels : numpy.array of strs
        A 1D numpy array containing names of chromosomes
    interchrom : bool
        Wether interchromosomal contacts should be considered
    sub_mats : list of scipy.sparse.csr_matrix
        Sub matrices extracted from the full contact map. Each matrix
        represents an inter or intra contact map.
    sub_mats_detectable_bins : list of numpy.array of ints
        Each element of the list is an array of detectable bin indices
        for the corresponding matrix in sub_mats.
    sub_mats_labels : numpy.array of strs
        Labels corresponding to the chromosomes in each sub matrix. Defaults
        to simple numbering scheme.
    bins : pandas.DataFrame
        Table containing bins genomic coordinates for the whole genome matrix.
    resolution : int
        Bin size of the Hi-C matrix.
    max_dist : int
        maximum distance from the diagonal at which intrachromosomal matrices
        should be scanned.
        """

    def __init__(self, mat_path, interchrom=False, max_dist=None):
        """
        Loads a cool or bg2 file containing a whole genome matrix, processes
        the contact map and splits it into intra and inter matrices.
        """
        self.interchrom = interchrom
        self.max_dist = max_dist
        # Load contacts, bin and chromosome infos from file
        self.mat, chromstarts, self.bins, self.resolution = self.load_data(mat_path)
        # Getting start and end coordinates of chromosomes into a 2D array
        chromend = np.append(chromstarts[1:], self.mat.shape[0])
        self.chroms = np.vstack([chromstarts, chromend]).T

        # Get indices of detectable bins and apply preprocessing treatment on matrix
        self.mat, self.detectable_bins = self.preprocess_full_matrix()

        # Splitting whole genome matrix into chromosome sub matrices
        self.sub_mats, self.sub_mats_detectable_bins = self.split_chromosomes()

        # TODO: Allow to read labels from file, for now we just use numbers
        self.sub_mats_labels = list(range(len(self.sub_mats)))

    def load_data(self, mat_path):
        """Load contact, bin and chromosome informations from input path"""
        # Define functions to use for each format
        format_loader = {"bg2": cio.load_bedgraph2d, "cool": cio.load_cool}
        # Guess file format fron file name
        extension = os.path.splitext(mat_path)[-1].lstrip(".")
        print("loading: ", mat_path)

        # Load contact map and chromosome start bins coords
        try:
            mat, chroms, bins, resolution = format_loader[extension](mat_path)
        except KeyError as e:
            sys.stderr.write(
                f"Unknown format: {extension}. Must be one of {format_loader.keys()}\n"
            )
            raise e

        return mat, chroms, bins, resolution

    def preprocess_full_matrix(self):
        """Apply general Hi-C preprocessing steps to the whole genome matrix"""
        detectable_bins = preproc.get_detectable_bins(self.mat)
        mat = preproc.normalize(self.mat, detectable_bins)
        return mat, detectable_bins

    def preprocess_intra_matrix(self, sub_mat, sub_mat_detectable_bins):
        # Remove speckles (outlier pixels)
        sub_mat = preproc.despeckle(sub_mat, th2=3)
        # Compute signal to noise ratio at all diagonals
        snr_dist = preproc.signal_to_noise_threshold(sub_mat, sub_mat_detectable_bins)
        # Define max_dist based on snr and pattern config
        sub_mat_max_dist = min(self.max_dist, snr_dist)
        # Detrend matrix for power law
        sub_mat = preproc.detrend(sub_mat, sub_mat_detectable_bins)
        # Remove pixels further than max_dist
        for diag in range(sub_mat_max_dist + 1, sub_mat.shape[0]):
            sub_mat.setdiag(0, diag)

        return sub_mat

    def preprocess_inter_matrix(self, sub_mat):
        return preproc.ztransform(sub_mat)

    def split_chromosomes(self):
        """
        Split the whole genome matrix into intra- and inter- chromosomal sub
        matrices. And apply specific preprocessing steps to sub-matrices.
        """
        # Convert whole genome matrix to CSR for indexing
        mat = self.mat.tocsr()
        # shorthand
        detect = self.detectable_bins
        sub_mats, sub_mats_detectable_bins = [], []
        # Loop over all possible combinations of chromosomes
        for start_c1, end_c1 in self.chroms:
            for start_c2, end_c2 in self.chroms:
                # Do not use lower triangle
                if start_c1 >= start_c2:
                    sub_mat = mat[start_c1:end_c1, start_c2:end_c2]
                    # Retrieve indices of detectable bins in submatrix and make
                    sub_mat_detectable_bins = (
                        detect[(detect >= start_c1) & (detect < end_c1)] - start_c1,
                        detect[(detect >= start_c2) & (detect < end_c2)] - start_c2,
                    )
                    # Intrachromosomal matrices need to be detrended for distance law
                    if start_c1 == start_c2:
                        sub_mat = self.preprocess_intra_matrix(
                            sub_mat, sub_mat_detectable_bins[0]
                        )
                        sub_mats.append(sub_mat)
                        sub_mats_detectable_bins.append(sub_mat_detectable_bins)
                    # But interchromsomal matrices must only be scaled
                    elif self.interchrom:
                        sub_mat = self.preprocess_inter_matrix(sub_mat)
                        sub_mats.append(sub_mat)
                        sub_mats_detectable_bins.append(sub_mat_detectable_bins)
        return sub_mats, sub_mats_detectable_bins

    def get_submat_idx(self, mat_type="intra"):
        """
        Retrieves indices of intra- or inter-chromosomal contact maps. This is
        done by extracting sub matrices that are on the diagonal of the 
        UPPER TRIANGLE full matrix.

        """
        n_chr = self.chroms.shape[0]
        # Retrieve indices of all matrices on the diagonal of the
        # upper triangle whole genome matrix.
        if self.interchrom:
            intra_mat_idx = [n_chr * i - (i ** 2 - i) / 2 for i in range(n_chr)]
        # Only intrachromosomal matrices have been kept, get all submatrices
        else:
            intra_mat_idx = list(range(n_chr))
        if mat_type == "inter":
            # Get all sub matrices that are not intra
            if self.interchrom:
                get_mat_idx = [
                    i for i in range(len(self.sub_mats)) if i not in intra_mat_idx
                ]
            # If interchrom is disabled, there is no inter- submatrix to get
            else:
                get_mat_idx = []
        else:
            get_mat_idx = intra_mat_idx

        return get_mat_idx

    def get_full_mat_pattern(self, pattern):
        """
        Converts bin indices of a pattern from an submatrix into their value in the
        original full-genome matrix.

        Parameters
        ----------
        pattern : tuple
            A pattern as given by explore_pattern (chrom, pos1, pos2, score).
            When using interchromosomal matrices, is an index based on the
            order in which submatrices where split.
        label : int
            The index of the submatrix in the list of submatrices. Depends on the
            order in which interchrom_wrapper split them.
        chroms : array_like
            2D numpy array containing start and end bins of chromosomes as columns,
            and one chromosome per row.
        """

        if pattern[1] == "NA":
            return pattern

        # Fancy trick to get chromosomes from matrix index in lower triangle of
        # whole genome matrix
        submat_idx = pattern[0]
        chrA = int(np.floor(-0.5 + np.sqrt(0.25 + 2 * submat_idx)))
        triangular_number = chrA * (chrA + 1) / 2
        chrB = int(submat_idx - triangular_number)
        # Get start bin for chromosomes of interest
        startA = self.chroms[chrA, 0]
        startB = self.chroms[chrB, 0]
        # Shift index by start bin of chromosomes
        inter_pattern = (0, pattern[1] + startA, pattern[2] + startB, pattern[3])
        return inter_pattern

