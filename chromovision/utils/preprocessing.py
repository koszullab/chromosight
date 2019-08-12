#!/usr/bin/env python3
# coding: utf-8

"""Preprocessing utils
Operations to perform on Hi-C matrices before analyses
"""
import numpy as np
from scipy.signal import savgol_filter


def normalize(B, good_bins=None, iterations=10):
    """
    Iterative normalisation of a Hi-C matrix.

    Parameters
    ----------
    B : scipy coo_matrix
        The Hi-C matrix to be normalized.
    good_bins : numpy array
        1D array containing the indices of detectable bins on which
        normalization should be applied.
        
    Returns
    -------
    numpy.ndarray :
        The SCN normalised Hi-C matrix
    """
    if good_bins is None:
        good_bins = np.arange(B.shape[0])
    r = B.copy()
    for _ in range(1, iterations):
        bin_sums = sum_mat_bins(r)
        # Retrieve coordinates of nonzero pixels
        pixel_rows, pixel_cols = r.nonzero()
        # Keep only coords of nonzero pixels that are in good col and row
        # Valid pixels will be nonzero values in good bins
        pixel_mask = np.isin(pixel_rows, good_bins) * np.isin(pixel_cols, good_bins)
        pixel_rows = pixel_rows[pixel_mask]
        pixel_cols = pixel_cols[pixel_mask]
        # ICE normalisation (Divide each valid pixel by the product of its row and column)
        r.data /= np.float64(bin_sums[pixel_rows] * bin_sums[pixel_cols])
    row_sums = np.array(r.sum(axis=1)).flatten()
    # Scale to 1
    r.data = r.data * (1 / np.mean(row_sums))
    return r


def distance_law(matrix, detectable_bins):
    """
    Computes genomic distance law by averaging over each diagonal in
    the upper triangle matrix.

    parameters
    ----------
    matrix: scipy.sparse.csr_matrix
        the input matrix to compute distance law from.
    detectable_bins : numpy.array of ints
        An array of detectable bins indices to consider when computing distance law.

    Returns
    -------
    dist: np.ndarray
        the output genomic distance law.

    example
    -------
        >>> m = np.ones((3,3))
        >>> m += np.array([1,2,3])
        >>> m
        array([[2., 3., 4.],
               [2., 3., 4.],
               [2., 3., 4.]])
        >>> distance_law(csr_matrix(m))
        array([3. , 3.5, 4. ])

    """
    n = min(matrix.shape)
    dist = np.zeros(n)
    for diag in range(n):
        dist[diag] = np.mean(matrix.diagonal(diag)[detectable_bins[: n - diag]])
    return dist


def despeckle(matrix, th2=3):
    """
    Remove speckles (i.e. noisy outlier pixels) from a Hi-C 
    contact map in sparse format. Speckles are set back to the
    median value of their respective diagonals.

    Parameters
    ----------

    B : scipy.sparse.coo_matrix
        Contact map in sparse upper triangle format.
    th2 : np.float64
        Threshold used for despeckling. This defines outlier pixel
        P on diagonal D by if P > median(D) + std(D) * th2

    Returns
    -------

    A : scipy.sparse.coo_matrix
        The despeckled sparse matrix.
    """
    matrix = matrix.tocoo()
    A = matrix.copy()
    n1 = A.shape[0]
    # Extract all diagonals in the upper triangle
    dist = {u: A.diagonal(u) for u in range(n1)}
    # Compute median and standard deviation for each diagonal
    medians, stds = {}, {}
    for u in dist:
        medians[u] = np.median(dist[u])
        stds[u] = np.std(dist[u])

    # Loop over all nonzero pixels in the COO matrix and their coordinates
    for i, (row, col, val) in enumerate(zip(matrix.row, matrix.col, matrix.data)):
        # Compute genomic distance of interactions in pixel
        dist = abs(row - col)
        # If pixel in input matrix is an outlier, set this pixel to median
        # of its diagonal in output matrix
        if val > medians[dist] + th2 * stds[dist]:
            A.data[i] = medians[dist]
    return A


def get_detectable_bins(matrix):
    """
    Returns lists of detectable indices after excluding low interacting bin
    based on the distribution of pixel values in the matrix.

    Parameters
    ----------
    matrix : array_like
        A Hi-C matrix in tihe form of a 2D numpy array or coo matrix

    Returns
    -------
    numpy array :
        1D array containing indices of low interacting bins.
    -------
    """
    sum_bins = sum_mat_bins(matrix)
    # Find poor interacting raws and columns
    threshold_bins = np.median(sum_bins) - 2.0 * np.std(sum_bins)
    # Removal of poor interacting rows and columns
    good_bins = np.where(sum_bins > threshold_bins)[0]
    return good_bins


def detrend(matrix, detectable_bins=None):
    """
    Detrends a Hi-C matrix by the distance law.
    The input matrix should have been normalised beforehandand.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The normalised intrachromosomal Hi-C matrix to detrend.
    detectable_bins : tuple
        Tuple containing a list of detectable rows and a list of columns on
        which to perform detrending. Poorly interacting indices have been
        excluded.

    Returns
    -------
    numpy.ndarray :
        The detrended Hi-C matrix.
    """
    matrix = matrix.tocsr()
    y = distance_law(matrix, detectable_bins[0])
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

    # Detrending by the distance law
    clean_mat = matrix.tocoo()
    clean_mat.data /= y_savgol[abs(clean_mat.row - clean_mat.col)]
    clean_mat = clean_mat.tocsr()
    # Set values in bad bins to 0
    miss_row_mask = np.ones(detectable_bins[0].shape, dtype=bool)
    miss_col_mask = np.ones(detectable_bins[1].shape, dtype=bool)
    miss_row_mask[detectable_bins[0]] = 0
    miss_col_mask[detectable_bins[1]] = 0
    clean_mat[np.ix_(miss_row_mask, miss_col_mask)] = 0.0
    clean_mat.eliminate_zeros()
    return clean_mat


def ztransform(matrix):
    """
    Z transformation for Hi-C matrices.

    Parameters
    ----------
    matrix : array_like
        A 2-dimensional numpy array Ms x Ns acting as a raw
        interchromosomal Hi-C map.

    Returns
    -------
    numpy.ndarray :
        A 2-dimensional numpy array of the z-transformed interchromosomal
        Hi-C map.
    """

    data = matrix.data
    mu = np.mean(data)
    sd = np.std(data)
    N = matrix.copy()
    N.data -= mu
    N.data /= sd

    return N


def sum_mat_bins(mat):
    """
    Compute the sum of matrices bins (i.e. rows or columns) using
    only the upper triangle, assuming symmetrical matrices.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Contact map in sparse format, either in upper triangle or
        full matrix.
    
    Returns
    -------
    numpy.array :
        1D array of bin sums.
    """
    # Equivalaent to row or col sum on a full matrix
    # Note: mat.sum returns a 'matrix' object. A1 extracts the 1D flat array
    # from the matrix
    return mat.sum(axis=0).A1 + mat.sum(axis=1).A1 - mat.diagonal(0)
