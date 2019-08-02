#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""
import sys
import numpy as np
from scipy.sparse import issparse, lil_matrix, csr_matrix
from scipy.ndimage import measurements
from scipy.signal import savgol_filter
import itertools

DEFAULT_PRECISION_CORR_THRESHOLD = 1e-4


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


def despeckles(B, th2):
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
    B = B.tocoo()
    A = B.copy()
    n1 = A.shape[0]
    # Extract all diagonals in the upper triangle
    dist = {u: A.diagonal(u) for u in range(n1)}
    # Compute median and standard deviation for each diagonal
    medians, stds = {}, {}
    for u in dist:
        medians[u] = np.median(dist[u])
        stds[u] = np.std(dist[u])

    # Loop over all nonzero pixels in the COO matrix and their coordinates
    for i, (row, col, val) in enumerate(zip(B.row, B.col, B.data)):
        # Compute genomic distance of interactions in pixel
        dist = abs(row - col)
        # If pixel in input matrix is an outlier, set this pixel to median
        # of its diagonal in output matrix
        if val > medians[dist] + th2 * stds[dist]:
            A.data[i] = medians[dist]
    return A


def picker(probas, thres=0.8):
    """Pick pixels out of a probability map
    Given a probability heat map, pick (i, j) of local maxima
    Parameters
    ----------
    probas : array_like in any sparse formats: it is converted in coo 
        A float array assigning a probability to each pixel (i,j)
        of being a loop.
    thres : float, optional
        Pixels having a probability higher than thres are potentially
        loops. Default is 0.8.
    Returns
    -------
    ijs : array_like
        Coordinates of identified loops.
    """
    # convertion from csr to coo:
    probas = probas.tocoo()
    # sanity check
    if np.any(probas.data > 1):
        raise ValueError("probas must be <= 1.0")
    if np.any(probas.data < 0):
        raise ValueError("probas must be >= 0.0")

    thres_mask = probas.data > thres
    rows_ijs = probas.row[thres_mask]
    col_ijs = probas.col[thres_mask]
    del thres_mask
    raw_ijs = np.array([rows_ijs, col_ijs]).T

    # sanity check
    if len(raw_ijs) > 0:
        I = max(raw_ijs[:, 0])
        J = max(raw_ijs[:, 1])
        candidate_p = np.zeros((I + 1, J + 1), bool)
        candidate_p[
            raw_ijs[:, 0], raw_ijs[:, 1]
        ] = True  #  heat map with foci of high proba
        labelled_mat, num_features = measurements.label(candidate_p)
        ijs = np.zeros([num_features, 2], int)
        remove_p = np.zeros(num_features, bool)
        # Iterate over candidate foci
        for ff in range(0, num_features):
            label_p = labelled_mat == ff + 1
            # remove the label corresponding to non-candidates
            if candidate_p[label_p].sum() == 0:
                remove_p[ff] = True
                continue
            # remove single points
            if label_p.sum() == 1:
                remove_p[ff] = True
                continue
            label_ijs = np.array(np.where(label_p)).T

            # conversion to lil format to have access to slicing
            probas = probas.tolil()
            ijmax = np.argmax(
                (probas[label_ijs[:, 0], label_ijs[:, 1]]).toarray()
            )
            ijs[ff, 0] = label_ijs[ijmax, 0]
            ijs[ff, 1] = label_ijs[ijmax, 1]
        ijs = ijs[~remove_p, :]
    else:
        ijs = "NA"
    return ijs


def picker_dense(probas, thres=0.8):
    """Pick pixels out of a probability map

    Given a probability heat map, pick (i, j) of local maxima

    Parameters
    ----------
    probas : array_like
        A float array assigning a probability to each pixel (i,j)
        of being a loop.
    thres : float, optional
        Pixels having a probability higher than thres are potentially
        loops. Default is 0.8.

    Returns
    -------
    ijs : array_like
        Coordinates of identified loops.
    """
    # sanity check
    if np.any(probas > 1):
        raise ValueError("probas must be <= 1.0")
    if np.any(probas < 0):
        raise ValueError("probas must be >= 0.0")

    raw_ijs = np.array(np.where(probas > thres)).T
    if len(raw_ijs) > 0:
        I = max(raw_ijs[:, 0])
        J = max(raw_ijs[:, 1])
        candidate_p = np.zeros((I + 1, J + 1), bool)
        candidate_p[
            raw_ijs[:, 0], raw_ijs[:, 1]
        ] = True  # heat map with foci of high proba
        labelled_mat, num_features = measurements.label(candidate_p)
        ijs = np.zeros([num_features, 2], int)
        remove_p = np.zeros(num_features, bool)
        for ff in range(0, num_features):
            label_p = labelled_mat == ff + 1
            # remove the label corresponding to non-candidates
            if candidate_p[label_p].sum() == 0:
                remove_p[ff] = True
                continue
            # remove single points
            if label_p.sum() == 1:
                remove_p[ff] = True
                continue
            label_ijs = np.array(np.where(label_p)).T
            ijmax = np.argmax(probas[label_ijs[:, 0], label_ijs[:, 1]])
            ijs[ff, 0] = label_ijs[ijmax, 0]
            ijs[ff, 1] = label_ijs[ijmax, 1]
        ijs = ijs[~remove_p, :]
    else:
        ijs = "NA"
    return ijs


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
    Detrends and removes speckles in a Hi-C matrix by the distance law.
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

    bin_sums = sum_mat_bins(matrix)

    # Removal of speckles (noisy pixels
    clean_mat = despeckles(matrix, 4.0)
    clean_mat = clean_mat.tocsr()

    y = distance_law(clean_mat, detectable_bins[0])
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

    n = matrix.shape[0]

    # Detrending by the distance law
    clean_mat = clean_mat.tocoo()
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


def xcorr2(
    signal,
    kernel,
    max_scan_distance=None,
    threshold=DEFAULT_PRECISION_CORR_THRESHOLD,
):
    """Signal-kernel 2D convolution

    Convolution of a 2-dimensional signal (the contact map) with a kernel
    (the pattern template).

    Parameters
    ----------
    signal: scipy.sparse.csr_matrix
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: array_like
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    max_scan_distance : int or None, optional
        Limits the range of computations beyond the diagonal. Default is None,
        which means no such limit is taken into account.
    threshold : float, optional
        Sets all values in the final matrix below this threshold to zero to
        reduce memory issues when handling sparse matrices. Default is set
        by the library, 1e-8.

    Returns
    -------
    out: scipy.sparse.csr_matrix
        2-dimensional numpy array that's the convolution product of signal
        by kernel. The shape of out depends on cenetred_p.
    """

    Ms, Ns = signal.shape
    Mk, Nk = kernel.shape

    if (Mk > Ms) or (Nk > Ns):
        raise ValueError("cannot have kernel bigger than signal")

    if max_scan_distance is None:
        max_scan_distance = max(Ms, Ns)

    Ki = (Mk - 1) // 2
    Kj = (Nk - 1) // 2
    out = lil_matrix((Ms, Ns))
    out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] = 0.0
    # Set a margin of kernel size below the diagonal to NA so
    # that id does not affect correlation
    for i in range(Mk):
        out.setdiag(0, -i)
    out = out.tocsr()

    for ki in range(Mk):
        # Note convolution is only computed up to a distance from the diagonal
        for kj in range(ki, min(Nk, ki + max_scan_distance)):
            out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] += (
                kernel[ki, kj]
                * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
            )
    out.eliminate_zeros()
    return out


def corrcoef2d(signal, kernel):
    """Signal-kernel 2D correlation

    Pearson correlation coefficient between signal and sliding kernel.
    """
    # Kernel1 allows to compute the mean
    kernel1 = np.ones(kernel.shape) / kernel.size
    # Returns a matrix of means
    mean_signal = xcorr2(signal, kernel1)
    std_signal = np.sqrt(xcorr2(signal ** 2, kernel1) - mean_signal ** 2)
    mean_kernel = np.mean(kernel)
    std_kernel = np.std(kernel)
    corrcoef = xcorr2(signal, kernel / kernel.size)
    # Compute correlation coefficient using the difference of logs
    # instead of the log of ratios since ratios of sparse matrices are
    # not possible
    # NOT GOOD: Find another way to do sparse matrix elementwise division
    numerator = (corrcoef - mean_signal * mean_kernel).log1p()
    denominator = (std_signal * std_kernel).log1p()
    corrcoef = numerator - denominator
    return corrcoef.expm1()




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
                #sub_mat = ztransform(sub_mat)
            # Only use lower triangle interchromosomal matrices
            elif s1 > s2 and interchrom:
                sub_mat = ztransform(sub_mat)
            else:
                continue
            # all submatrices are ztransformed to get same scale
            matrices.append(sub_mat)
            vectors.append(sub_mat_idx)

    return matrices, vectors

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
