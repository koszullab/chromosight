#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""
import sys
import numpy as np
from scipy.sparse import find
from scipy.ndimage import measurements
from scipy.signal import savgol_filter
import itertools
from copy import copy


def scn_func(B, mat_idx=None):
    """
    Sequential Component Normalisation (SCN) of a Hi-C matrix.

    Parameters
    ----------
    B : array_like
        The Hi-C matrix to be normalised.
    mat_idx : tuple
        A tuple of 2 lists, containing the indices of detectable rows and
        columns on which SCN should be applied.
        
    Returns
    -------
    numpy.ndarray :
        The SCN normalised Hi-C matrix
    """
    try:
        m_format = B.getformat()  # raises an  AttributeError if matrix is dense
        A = B.copy()
        A = A.tolil()
    except:
        if isinstance(B, np.ndarray):
            m_format = "dense"
            A = np.copy(B)
        else:
            sys.stderr.write(
                "ERROR: the matrix to normalize is neither dense or sparse."
            )
            sys.exit(1)
    finally:
        nr = A.shape[0]
        nc = A.shape[1]
        # If no list of bins is specified, include all bins
        if not mat_idx:
            mat_idx = (np.arange(nr), np.arange(nc))
        mask_r = np.zeros(nr, dtype=bool)
        mask_c = np.zeros(nc, dtype=bool)
        mask_r[mat_idx[0]] = 1
        mask_c[mat_idx[1]] = 1
        n_iterations = 10
        for _ in range(n_iterations):
            for i in range(nc):
                A[mask_r, i] = A[mask_r, i] / np.sum(A[mask_r, i])
                A[~mask_r, i] = 0
                A.data = np.nan_to_num(A.data)
            for i in range(nr):
                A[i, mask_c] = A[i, mask_c] / np.sum(A[i, mask_c])
                A[i, ~mask_c] = 0
                A.data = np.nan_to_num(A.data)
    if m_format != "dense":
        A = A.tocoo()
        A.eliminate_zeros()
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
    A = np.copy(B)
    n_speckles = 0
    outlier = []
    n1 = A.shape[0]
    dist = {u: np.diag(A, u) for u in range(n1)}

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


def get_mat_idx(matrix):
    """
    Returns lists of detectable indices after excluding low interacting bin
    based on the distribution of pixel values in the matrix.
    
    Parameters
    ----------
    matrix : array_like
        A Hi-C matrix in tihe form of a 2D numpy array or coo matrix

    Returns
    -------
    tuple :
        A tuple of two 1D arrays containing indices of low interacting rows
        and columns, respectively.
    -------
    """
    try:
        m_format = matrix.getformat()  # raises an  AttributeError if matrix is dense
        matrix = matrix.tolil()
    except:
        if not isinstance(B, np.ndarray):
            sys.stderr.write(
                "ERROR: the matrix to get idx from is neither dense or sparse."
            )
            sys.exit(1)
    finally:
        # Sum raws and columns
        sum_axis0 = np.array(matrix.sum(axis=0))
        to_reshape = sum_axis0.shape
        sum_axis1 = np.array(matrix.sum(axis=1)).reshape(to_reshape)
        # Find poor interacting raws and columns
        threshold_rows = np.median(sum_axis0) - 2.0 * np.std(sum_axis0)
        threshold_cols = np.median(sum_axis1) - 2.0 * np.std(sum_axis1)
        # Removal of poor interacting rows and columns
        ind_rows = np.where(sum_axis0 > threshold_rows)[0]
        ind_cols = np.where(sum_axis1 > threshold_cols)[0]
        good_bins = (ind_rows, ind_cols)
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
    if matrix.shape[0] != matrix.shape[1] or len(mat_idx[0]) != len(mat_idx[1]):
        raise ValueError("Detrending can only be done on square matrices.")

    n = matrix.shape[0]
    if mat_idx:
        poor_idx = [i for i in range(n) if i not in mat_idx[0]]
    else:
        poor_idx = []

    y = distance_law(matrix)
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

    # Computation of genomic distance law matrice:
    distance_law_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_law_matrix[i, j] = y_savgol[abs(j - i)]
    detrended = matrix / distance_law_matrix
    detrended[np.isnan(detrended)] = 1.0
    detrended[detrended < 0] = 1.0
    # refilling of empty bins with 1.0 (neutral):
    detrended[poor_idx, :] = np.ones((len(poor_idx), n))
    detrended[:, poor_idx] = np.ones((n, len(poor_idx)))
    return detrended


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

    mu = np.mean(matrix)
    sd = np.std(matrix)
    N = np.copy(matrix)
    N = (N - mu) / sd
    return N


def xcorr2(signal, kernel, centered_p=True):
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

    if not (centered_p):
        out = np.zeros((Ms - Mk + 1, Ns - Nk + 1))
        for ki in range(Mk):
            for kj in range(Nk):
                out += (
                    kernel[ki, kj]
                    * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
                )
    else:
        Ki = (Mk - 1) // 2
        Kj = (Nk - 1) // 2
        out = np.zeros((Ms, Ns)) + np.nan
        out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] = 0.0
        for ki in range(Mk):
            for kj in range(Nk):
                out[Ki : Ms - (Mk - 1 - Ki), Kj : Ns - (Nk - 1 - Kj)] += (
                    kernel[ki, kj]
                    * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
                )

    return out


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
        using interchromosomal matrices, chrom represents the order in which sub
        matrices where split.
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
    mat_idx = get_mat_idx(matrix)
    matrix = scn_func(matrix, mat_idx)
    for s1, e1 in chroms:
        for s2, e2 in chroms:
            sub_mat = matrix[s1:e1, s2:e2]
            # Get new indices of low interacting bins within sub matrix
            sub_mat_idx = (
                mat_idx[0][(mat_idx[0] >= s1) & (mat_idx[0] < e1)] - s1,
                mat_idx[1][(mat_idx[1] >= s2) & (mat_idx[1] < e2)] - s2,
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
