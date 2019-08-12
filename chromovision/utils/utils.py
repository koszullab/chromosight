#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components


def interchrom_wrapper(matrix, chroms, interchrom=False):
    """
    Given a matrix containing multiple chromosomes, processes each
    inter- or intra-chromosomal submatrix to be chromovision-ready.
    Given 1 matrix and N chromosomes, N matrices will be returned if
    interchrom is False, or ((N^2)+N)/2 matrices otherwise (i.e. inter-
    matrices in the upper triangle + intra- matrices).

    Parameters
    ----------
    matrix : array_like
        A 2D numpy array containing the whole Hi-C matrix made of multiple
        chromosomes
    chromstart : array_like
        A 2D numpy array containing with start and end bins of chromosomes,
        as columns and 1 chromosome per row.
    interchrom : bool
        Wether interchromosomal contacts should be conserved. Defaults to
        False, discarding interchromosomal matrices.

    Returns
    -------
    matrices : list of numpy array
        list of 2D numpy arrays containing intra or interchromosomal
        matrices. Each intra- or inter-chromosomal sub-matrix is detrended
        or z-transformed.
    vectors : list of numpy array
        For each matrix, a corresponding array of detectable bins indices.
    """
    matrices = []
    vectors = []
    mat_idx = get_detectable_bins(matrix)
    matrix = normalize(matrix, mat_idx)
    matrix = matrix.tocsr()
    for s1, e1 in chroms:
        for s2, e2 in chroms:
            sub_mat = matrix[s1:e1, s2:e2]
            # Get new indices of low interacting bins within sub matrix
            sub_mat_idx = (
                mat_idx[(mat_idx >= s1) & (mat_idx < e1)] - s1,
                mat_idx[(mat_idx >= s2) & (mat_idx < e2)] - s2,
            )
            # intrachromosomal sub matrix
            if s1 == s2:
                sub_mat = detrend(sub_mat, sub_mat_idx)
                # sub_mat = ztransform(sub_mat)
            # Only use lower triangle interchromosomal matrices
            elif s1 > s2 and interchrom:
                sub_mat = ztransform(sub_mat)
            else:
                continue
            # all submatrices are ztransformed to get same scale
            matrices.append(sub_mat)
            vectors.append(sub_mat_idx)

    return matrices, vectors

