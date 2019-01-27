#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""

import numpy as np
from scipy.ndimage import measurements
from scipy.signal import savgol_filter
import itertools


def scn_func(B, threshold=0):
    A = np.copy(B)
    nr = A.shape[0]
    nc = A.shape[1]
    n_iterations = 10
    keepr = np.zeros((nr, 1))
    keepc = np.zeros((nc, 1))

    for i in range(nr):
        if np.sum(A[i, :]) > threshold:
            keepr[i] = 1
        else:
            keepr[i] = 0

    for i in range(nc):
        if np.sum(A[:, i]) > threshold:
            keepc[i] = 1
        else:
            keepc[i] = 0

    ind_r_keep = np.where(keepr > 0)
    ind_r_drop = np.where(keepr <= 0)
    ind_c_keep = np.where(keepc > 0)
    ind_c_drop = np.where(keepc <= 0)

    for _ in range(n_iterations):
        for i in range(nc):
            A[ind_r_keep[0], i] = A[ind_r_keep[0], i] / \
                                  np.sum(A[ind_r_keep[0], i])
            A[ind_r_drop[0], i] = 0
        A[np.isnan(A)] = 0.0

        for i in range(nr):
            A[i, ind_c_keep[0]] = A[i, ind_c_keep[0]] / \
                                  np.sum(A[i, ind_c_keep[0]])
            A[i, ind_c_drop[0]] = 0
        A[np.isnan(A)] = 0.0
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
        dist[diag] = np.mean(np.diag(matrix, -diag))
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


def detrend(matrix):
    threshold_vector = np.median(matrix.sum(axis=0)) - 2.0 * np.std(
        matrix.sum(axis=0)
    )  # Removal of poor interacting bins
    poor_indices = np.where(matrix.sum(axis=0) <= threshold_vector)
    matscn = scn_func(matrix, threshold_vector)
    _, matscn, _, _ = despeckles(matscn, 10.0)

    y = distance_law(matscn)
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

    n = matrix.shape[0]

    # Computation of genomic distance law matrice:
    distance_law_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_law_matrix[i, j] = y_savgol[abs(j - i)]
    detrended = matscn / distance_law_matrix
    detrended[np.isnan(detrended)] = 1.0
    detrended[detrended < 0] = 1.0
    # refilling of empty bins with 1.0 (neutral):
    detrended[poor_indices[0], :] = np.ones((len(poor_indices[0]), n))
    detrended[:, poor_indices[0]] = np.ones((n, len(poor_indices[0])))
    return detrended, threshold_vector


def ztransform(matrix):
    """
    Z transformation for interchromosomal matrices.
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
    print(matrix.shape)
    threshold_vector = np.median(matrix.sum(axis=0)) - 2.0 * np.std(
        matrix.sum(axis=0)
    )  # Removal of poor interacting bins
    print(threshold_vector)
    matscn = scn_func(matrix, threshold_vector)
    _, matscn, _, _ = despeckles(matscn, 10.0)

    mu = np.mean(matscn)
    sd = np.sd(matscn)
    return (matscn - mu) / sd, threshold_vector


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
    std_signal = np.sqrt(
        xcorr2(signal ** 2, kernel1, centered_p) - mean_signal ** 2
    )
    mean_kernel = np.mean(kernel)
    std_kernel = np.std(kernel)
    corrcoef = (
        xcorr2(signal, kernel / kernel.size, centered_p)
        - mean_signal * mean_kernel
    ) / (std_signal * std_kernel)
    return corrcoef


def interchrom_wrapper(matrix, chromstart):
    """
    Given a matrix containing multiple chromosomes, processes each
    inter- or intra-chromosomal submatrix to be chromovision-ready.
    Parameters
    ----------
    matrix : array_like
        A 2D numpy array containing the whole Hi-C matrix made of multiple
        chromosomes
    chromstart : array_like
        A 1D numpy array containing the start bins of chromosomes,
        as intervals of bins.
    Returns
    array_like :
        A 2D numpy array containing the whole processed matrix. Each
        intra- or inter-chromosomal sub-matrix is detrended or z-transformed.
    -------
    """
    matrices = []
    vectors = []
    chromend = np.append(chromstart[1:], matrix.shape[0])
    chroms = np.vstack([chromstart, chromend]).T
    for s1, e1 in chroms:
        for s2, e2 in chroms:
            # intrachromosomal sub matrix
            if s1 == s2:
                tmp = detrend(matrix[s1:e1, s2:e2])
            else:
                tmp = ztransform(matrix[s1:e1, s2:e2])
            matrices.append(tmp[0])
            vectors.append(tmp[1])

    return matrices, vectors
