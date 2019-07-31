# Implementation of a contact map class.
import chromovision.utils as utils
import numpy as np

class ContactMap(object):

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
    interchrom : bool
        Wether interchromosomal contacts should be considered
    chrom_labels : numpy.array of strs
        A 1D numpy array containing names of chromosomes
    sub_mats : list of scipy.sparse.csr_matrix
        Sub matrices extracted from the full contact map. Each matrix
        represents an inter or intra contact map.
    sub_mats_detectable_bins : list of numpy.array of ints
        Each element of the list is an array of detectable bin indices
        for the corresponding matrix in sub_mats.
    sub_mats_labels : numpy.array of strs
        Labels corresponding to the chromosomes in each sub matrix. Defaults
        to simple numbering scheme.
        """

    def __init__(self, mat, chroms, interchrom=False):
        self.mat = mat
        self.interchrom = interchrom
        # Getting start and end coordinates of chromosomes
        chromend = np.append(chroms[1:], mat.shape[0])
        # 
        self.chroms = np.vstack([chroms, chromend]).T
        self.sub_mats, self.sub_mats_detectable_bins = utils.interchrom_wrapper(
                mat,
                self.chroms,
                interchrom
        )
        self.sub_mats_labels = list(range(len(self.sub_mats)))
    
    def get_intra_sub_mats(self):
        """
        Retrieves indices for intrachromosomal contact maps. This is done by
        extracting sub matrices that are on the diagonal of the UPPER TRIANGLE 
        full matrix.

        """
        n_chr = self.chroms.shape[0]
        if self.interchrom:
            return self.sub_mats
        return [self.sub_mats[n_chr * i - (i**2 - i) / 2] for i in range(n_chr)]

    def get_inter_sub_mats(self):
        """Retrieves interchromosomal sub matrices"""
        if self.interchrom:
            n_chr = len(self.chroms.shape[0])
            # Retrieve all matrices except those on the diagonal of the
            # full upper triangle matrix.
            intra_mat_idx = [n_chr * i - (i**2 - i) / 2 for i in range(n_chr)]
            return [self.sub_mats[i] for i in range(n_chr) if i not in intra_mat_idx]
        return None

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

