#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""
import sys
import numpy as np
from scipy.sparse import issparse, lil_matrix
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
    if not good_bins:
        good_bins = np.arange(B.shape[0])
    if issparse(B):
        A = B.copy()
        A = A.tolil()
        # Making full symmetric matrix if not symmetric already (e.g. upper triangle)
        r = r.tocoo()
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
            r.data /= bin_sums[pixel_rows] * bin_sums[pixel_cols]
        row_sums = np.array(r.sum(axis=1)).flatten()
        # Scale to 1
        r.data = r.data * (1 / np.mean(row_sums))
    else:
        A = np.copy(B)
        # If no list of bins is specified, include all bins
        bin_mask = np.zeros(A.shape[0], dtype=bool)
        bin_mask[good_bins] = 1
        for _ in range(0, iterations):
            # Compute matrix with all products of rows / cols sums
            mat_sum_prods = np.dot(A[:, None], A[:, None].T)
            # Element wise distribution by pixel's respective row/col sum product
            mat_sum_prods[~bin_mask, :] = 1
            mat_sum_prods[:, ~bin_mask] = 1
            A[bin_mask, bin_mask] /= mat_sum_prods[bin_mask, bin_mask]
    return A


def distance_law(matrix):
    """Genomic distance law

    Compute genomic distance law by averaging over each diagonal.

    Parameters
    ----------
    matrix: array_like
        The input matrix to compute distance law from.

    Returns
    -------
    dist: np.ndarray
        The output genomic distance law.

    Example
    -------
        >>> M = np.ones((3,3))
        >>> M += np.array([1,2,3])
        >>> M
        array([[2., 3., 4.],
               [2., 3., 4.],
               [2., 3., 4.]])
        >>> distance_law(M)
        array([3. , 2.5, 2. ])

    """
    n = matrix.shape[0]
    dist = np.zeros(n)
    for diag in range(n):
        dist[diag] = np.mean(np.array(matrix.diagonal(-diag)))
    return dist


def despeckles(B, th2):

    n_speckles = 0
    n1 = 0
    outlier = []
    dist = dict()
    A = np.copy(B)
    if isinstance(B, np.ndarray):
        A = np.copy(B)
        n1 = A.shape[0]
        dist = {u: np.diag(A, u) for u in range(n1)}
    elif issparse(B):

        A = B.copy()
        n1 = A.shape[0]
        matrix_format = A.getformat()
        if matrix_format not in {"csr", "csc"}:
            A = A.tocsr()
        dist = {u: A.diagonal(u) for u in range(n1)}

    medians, stds = {}, {}
    for u in dist:
        medians[u] = np.median(dist[u])
        stds[u] = np.std(dist[u])

    for nw, j in itertools.product(range(n1), range(n1)):
        lp = j + nw
        kp = j - nw
        if lp < n1:
            if A[j, lp] > medians[nw] + th2 * stds[nw]:
                A[j, lp] = medians[nw]
                n_speckles += 1
                outlier.append((j, lp))
        if kp >= 0:
            if A[j, kp] > medians[nw] + th2 * stds[nw]:
                A[j, kp] = medians[nw]
                n_speckles += 1
                outlier.append((j, kp))
    return dist, A, n_speckles, outlier


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

    rows_ijs = probas.row[probas.data > thres]
    col_ijs = probas.col[probas.data > thres]
    raw_ijs = np.array([rows_ijs, col_ijs]).T

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


def detrend(matrix, mat_idx=None):
    """
    Detrending a Hi-C matrix by the distance law. The matrix should have been
    normalised using the SCN procedure beforehandand then detrended by the
    distance law.

    Parameters
    ----------
    matrix : array_like
        The intrachromosomal Hi-C matrix to detrend
    mat_indices : tuple
        Tuple containing a list of detectable rows and a list of columns on
        which to perform detrending. Poorly interacting indices have been
        excluded.

    Returns
    -------
    numpy.ndarray :
        The detrended Hi-C matrix.
    tuple :
        Tuple of thresholds to define low interacting rows/columns.
    """
    summed_columns = np.array(
        np.array(matrix.sum(axis=0)).flatten()
        + np.array(matrix.sum(axis=1)).flatten()
        - matrix.diagonal()
    )
    threshold = np.median(summed_columns) - 2.0 * np.std(summed_columns)

    # Removal of poor interacting bins
    poor_indices = np.where(summed_columns <= threshold)
    matscn = scn_func(matrix, threshold)
    _, matscn, _, _ = despeckles(matscn, 10.0)

    y = distance_law(matscn)
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

    n = matrix.shape[0]

    # Computation of genomic distance law matrice:
    if issparse(matrix):
        distance_law_matrix = sparse.csr_matrix((n, n))
        distance_law_matrix = sparse.triu(distance_law_matrix)
    else:
        distance_law_matrix = np.zeros((n, n))
        distance_law_matrix = np.triu(distance_law_matrix)
    for i in range(0, n):
        for j in range(i, n):
            distance_law_matrix[i, j] = y_savgol[abs(j - i)]
    if issparse(matrix):
        detrended = sparse.csr_matrix((n, n))
        for i in range(0, 2):
            for j in range(i, n):
                val = matscn[i, j] / distance_law_matrix[i, j]
                if val != np.nan:
                    if val > 0:
                        detrended[i, j] = matscn[i, j] / distance_law_matrix[i, j]
                    else:
                        detrended[i, j] = 1
    else:
        detrended = matscn / distance_law_matrix
        detrended[np.isnan(detrended)] = 1.0
        detrended[detrended < 0] = 1.0
    # refilling of empty bins with 1.0 (neutral):
    detrended[poor_indices[0], :] = np.ones((len(poor_indices[0]), n))
    detrended[:, poor_indices[0]] = np.ones((n, len(poor_indices[0])))
    return detrended, threshold


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
    tuple of floats :
        A tuple of two floats containing the threshold for low interacting
        rows and cols, respectively.
    """

    if isinstance(matrix, np.ndarray):
        mu = np.mean(matrix)
        sd = np.std(matrix)
        N = np.copy(matrix)
        N = (N - mu) / sd

    elif issparse(matrix):
        data = matrix.data
        mu = np.mean(data)
        sd = np.std(data)
        N = matrix.copy()
        N.data -= mu
        N.data /= sd

    else:
        raise ValueError("Matrix must be in sparse or dense format.")

    return N


def xcorr2(
    signal,
    kernel,
    centered_p=True,
    max_size=None,
    threshold=DEFAULT_PRECISION_CORR_THRESHOLD,
):
    """Signal-kernel 2D convolution

    Convolution of a 2-diemensional signal (the contact map) with a kernel
    (the pattern template).

    Parameters
    ----------
    signal: array_like
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: array_like
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    centered_p: bool, optional
        If False, then return a matrix with shape (Ms-Mk+1) x (Ns-Nk+1),
        otherwise return a matrix with shape Ms x Ns, with values located at
        center of kernel. Default is True.
    max_size : int or None, optional
        Limits the range of computations beyond the diagonal. Default is None,
        which means no such limit is taken into account.
    threshold : float, optional
        Sets all values in the final matrix below this threshold to zero to
        reduce memory issues when handling sparse matrices. Default is set
        by the library, 1e-8.

    Returns
    -------
    out: numpy.ndarray
        2-dimensional numpy array that's the convolution product of signal
        by kernel. The shape of out depends on cenetred_p.
    """

    Ms, Ns = signal.shape
    Mk, Nk = kernel.shape

    if (Mk > Ms) or (Nk > Ns):
        raise ValueError("cannot have kernel bigger than signal")

    if max_size is None:
        max_size = max(Ms, Ns)

    if not centered_p:
        out = np.zeros((Ms - Mk + 1, Ns - Nk + 1))
        for ki in range(Mk):
            for kj in range(ki, max(Nk, ki + max_size)):
                out += (
                    kernel[ki, kj]
                    * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
                )
    else:
        Ki = (Mk - 1) // 2
        Kj = (Nk - 1) // 2
        out = lil_matrix((Ms, Ns))
        out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] = 0.0
        for ki in range(Mk):
            for kj in range(ki, max(Nk, ki + max_size)):
                out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] += (
                    kernel[ki, kj]
                    * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
                )

    return out.tocsr()


def corrcoef2d(signal, kernel, centered_p=True):
    """Signal-kernel 2D correlation

    Pearson correlation coefficient between signal and sliding kernel.
    """
    kernel1 = np.ones(kernel.shape) / kernel.size
    mean_signal = xcorr2(signal, kernel1, centered_p)
    std_signal = np.sqrt(xcorr2(signal ** 2, kernel1, centered_p) - mean_signal ** 2)
    mean_kernel = np.mean(kernel)
    std_kernel = np.std(kernel)
    corrcoef = (
        xcorr2(signal, kernel / kernel.size, centered_p) - mean_signal * mean_kernel
    ) / (std_signal * std_kernel)
    return corrcoef


def get_inter_idx(pattern, chroms):
    """
    Converts bin indices of a pattern from an submatrix into their value in the
    original full-genome matrix.

    Parameters
    ----------
    pattern : tuple
        A pattern as given by explore_pattern (chrom, pos1, pos2, score). When
        using interchromosomal matrices, chrom represents the order in which
        submatrices where split.
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
    startA = chroms[chrA, 0]
    startB = chroms[chrB, 0]
    # Shift index by start bin of chromosomes
    inter_pattern = (0, pattern[1] + startA, pattern[2] + startB, pattern[3])
    return inter_pattern


def interchrom_wrapper(matrix, chroms):
    """
    Given a matrix containing multiple chromosomes, processes each
    inter- or intra-chromosomal submatrix to be chromovision-ready.

    Parameters
    ----------
    matrix : array_like
        A 2D numpy array containing the whole Hi-C matrix made of multiple
        chromosomes
    chromstart : array_like
        A 2D numpy array containing with start and end bins of chromosomes,
        as columns and 1 chromosome per row.

    Returns
    -------
    array_like :
        A 2D numpy array containing the whole processed matrix. Each
        intra- or inter-chromosomal sub-matrix is detrended or z-transformed.
    """
    matrices = []
    vectors = []
    mat_idx = get_detectable_bins(matrix)
    matrix = scn_func(matrix, mat_idx)
    for s1, e1 in chroms:
        for s2, e2 in chroms:
            sub_mat = matrix[s1:e1, s2:e2]
            # Get new indices of low interacting bins within sub matrix
            sub_mat_idx = (
                mat_idx[0][(mat_idx >= s1) & (mat_idx < e1)] - s1,
                mat_idx[1][(mat_idx >= s2) & (mat_idx < e2)] - s2,
            )
            # intrachromosomal sub matrix
            if s1 == s2:
                detrended = detrend(sub_mat, sub_mat_idx)
                sub_mat = detrended
                sub_mat = ztransform(detrended)
            # Only use lower triangle matrices
            elif s1 > s2:
                # sub_mat = ztransform(sub_mat)
                pass
            else:
                continue
            # all submatrices are ztransformed to get same scale
            matrices.append(sub_mat)
            vectors.append(sub_mat_idx)

    return matrices, vectors
