#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and 
loop/border data.
"""

import numpy as np
from scipy.ndimage import measurements
from scipy.signal import savgol_filter


def scn_func(A, threshold=0):
    n1 = A.shape[0]
    n_iterations = 10
    keep = np.zeros((n1, 1))

    for i in range(0, n1):
        if np.sum(A[i,]) > threshold:
            keep[i] = 1
        else:
            keep[i] = 0

    indices1 = np.where(keep > 0)
    indices2 = np.where(keep <= 0)

    for n in range(0, n_iterations):
        for i in range(0, n1):
            A[indices1[0], i] = A[indices1[0], i] / np.sum(A[indices1[0], i])
            A[indices2[0], i] = 0
        A[np.isnan(A)] = 0.0

        for i in range(0, n1):
            A[i, indices1[0]] = A[i, indices1[0]] / np.sum(A[i, indices1[0]])
            A[i, indices2[0]] = 0
        A[np.isnan(A)] = 0.0
    return A


def distance_law(A):
    n1 = A.shape[0]
    dist = np.zeros(n1)
    for nw in range(n1):
        dist[nw] = np.mean(np.diag(A, -nw))
    return dist


def distance_law_filter(A, indices):
    n1 = A.shape[0]
    dist = np.zeros(n1)
    n_int = np.zeros(n1)
    for nw in range(n1):  # scales
        group = []
        for j in range(0, n1):
            lp = j + nw
            if j in indices and lp in indices and lp <= n1:
                group.append(A[j, lp])
        dist[nw] = np.mean(group)
    return dist


def despeckles(A, th2):
    n_speckles = 0
    outlier = []
    n1 = A.shape[0]
    dist = {}
    n_int = np.zeros(n1)
    for nw in range(n1):  # scales
        group = []
        for j in range(0, n1):
            lp = j + nw
            if lp < n1:
                group.append(A[j, lp])
        dist[nw] = group

    for nw in range(n1):  # scales
        for j in range(0, n1):  # along the chromosome
            lp = j + nw
            kp = j - nw
            if lp < n1:
                if A[j, lp] > np.median(dist[nw]) + th2 * np.std(dist[nw]):
                    A[j, lp] = 0
                    n_speckles += 1
                    outlier.append((j, lp))
            if kp >= 0:
                if A[j, kp] > np.median(dist[nw]) + th2 * np.std(dist[nw]):
                    A[j, kp] = 0
                    n_speckles += 1
                    outlier.append((j, kp))
    return dist, A, n_speckles, outlier


def picker(probas, thres=0.8):
    """
    Given a probability heat map, pick (i, j) of local maxima
    INPUT:
    probas: a float array assigning a probability to each pixel (i,j)
            of being a loop.
    thres:  pixels having a probability higher than thres are potentially
            loops (default is 0.8).
    OUTPUT:
    ijs: coordinates of identified loops.
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
            ijmax = np.argmax(probas[label_ijs[:, 0], label_ijs[:, 1]])
            ijs[ff, 0] = label_ijs[ijmax, 0]
            ijs[ff, 1] = label_ijs[ijmax, 1]
        ijs = ijs[~remove_p, :]
    else:
        ijs = "NA"
    return ijs


def loop_detector(list_files, p, precision_value=4.0):
    LIST_SIZES = []  # list of sizes of detected loops
    MAT_LIST = []  # list containing all pannel of detected patterns
    area = 8  # Half size of the pannel
    MAT_SUM = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # sum of all detected patterns
    MAT_MEDIAN = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # median of all detected patterns
    loops_peak_selected = []
    n_patterns = 0
    for fich in list_files:
        print(fich)
        chromo = fich
        #        chromo = "chr"+re.findall("chr""([0-9]+)", fich)[0]
        print(chromo)
        raw_test = np.loadtxt(fich)
        th_sum = np.median(raw_test.sum(axis=0)) - 2.0 * np.std(
            raw_test.sum(axis=0)
        )  # Removal of poor interacting bins
        indices_matrice = np.where(raw_test.sum(axis=0) > th_sum)
        indices_poor = np.where(raw_test.sum(axis=0) <= th_sum)
        matscn = scn_func(raw_test, th_sum)
        dd, matscn, n, outiers = despeckles(matscn, 3.0)

        dist = distance_law(matscn)
        x = np.arange(0, len(dist))
        y = dist
        y[np.isnan(y)] = 0.
        y_savgol = savgol_filter(y, window_length=17, polyorder=5)

        n1 = np.shape(matscn)[0]
        # Computation of genomic distance law matrice:
        MAT_DIST = np.zeros((n1, n1))
        for i in range(0, n1):
            for j in range(0, n1):
                MAT_DIST[i, j] = y_savgol[abs(j - i)]
        MAT_DETREND = matscn / MAT_DIST
        MAT_DETREND[np.isnan(MAT_DETREND)] = 1.0
        MAT_DETREND[MAT_DETREND < 0] = 1.0
        # refilling of empty bins with 1.0 (neutral):
        for i in indices_matrice[0]:
            MAT_DETREND[indices_poor[0], :] = np.ones(
                (len(indices_poor[0]), n1)
            )
            MAT_DETREND[:, indices_poor[0]] = np.ones(
                (n1, len(indices_poor[0]))
            )

        loops_peak_selected_chromo = []
        res2 = corrcoef2d(
            MAT_DETREND, p, centered_p=False
        )  # !!  Here the pattern match  !!
        res2[np.isnan(res2)] = 0.0
        n2 = np.shape(res2)[0]
        res_rescaled = np.zeros(np.shape(matscn))
        res_rescaled[
            np.ix_(
                range(int(area), n2 + int(area)),
                range(int(area), n2 + int(area)),
            )
        ] = res2
        VECT_VALUES = np.reshape(res_rescaled, (1, n1 * n1))
        VECT_VALUES = VECT_VALUES[0]
        thr = np.median(VECT_VALUES) + precision_value * np.std(VECT_VALUES)
        indices_max = np.where(res_rescaled > thr)
        indices_max = np.array(indices_max)
        res_rescaled = np.triu(res_rescaled)  # Centering:
        res_rescaled[(res_rescaled) < 0] = 0
        loops_peak = picker(res_rescaled, thr)  #   Recentring here !!

        if loops_peak != "NA":
            mask = np.array(abs(loops_peak[:, 0] - loops_peak[:, 1])) < 5000
            loops_peak = loops_peak[mask, :]
            mask = np.array(abs(loops_peak[:, 0] - loops_peak[:, 1])) > 2
            loops_peak = loops_peak[mask, :]
            for l in loops_peak:
                if l[0] in indices_matrice[0] and l[1] in indices_matrice[0]:
                    p1 = int(l[0])
                    p2 = int(l[1])
                    if p1 > p2:
                        p22 = p2
                        p2 = p1
                        p1 = p22
                    if (
                        p1 - area >= 0
                        and p1 + area + 1 < n1
                        and p2 - area >= 0
                        and p2 + area + 1 < n1
                    ):
                        MAT_PANNEL = MAT_DETREND[
                            np.ix_(
                                range(p1 - area, p1 + area + 1),
                                range(p2 - area, p2 + area + 1),
                            )
                        ]
                        if (
                            len(MAT_PANNEL[MAT_PANNEL == 1.])
                            < ((area * 2 + 1) ** 2) * 1.0 / 100.
                        ):  # there should not be many indetermined bins
                            n_patterns += 1
                            score = res_rescaled[l[0], l[1]]
                            loops_peak_selected.append(
                                [chromo, l[0], l[1], score]
                            )
                            MAT_SUM = MAT_SUM + MAT_PANNEL
                            MAT_LIST.append(MAT_PANNEL)
                            LIST_SIZES.append(abs(p2 - p1))
                        else:
                            loops_peak_selected.append(
                                [chromo, "NA", "NA", "NA"]
                            )
        else:
            loops_peak_selected.append([chromo, "NA", "NA", "NA"])

    # Computation of stats on the whole set - Agglomerated procedure :
    for i in range(0, area * 2 + 1):
        for j in range(0, area * 2 + 1):
            list_temp = []
            for el in range(1, len(MAT_LIST)):
                list_temp.append(MAT_LIST[el][i, j])
            MAT_MEDIAN[i, j] = np.median(list_temp)

    return loops_peak_selected, LIST_SIZES, MAT_LIST, MAT_MEDIAN


def border_detector(list_files, p, precision_value=4.0):
    LIST_SIZES = []  # list of sizes of detected loops
    MAT_LIST = []  # list containing all pannel of detected patterns
    area = 8  # Half size of the pannel
    MAT_SUM = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # sum of all detected patterns
    MAT_MEDIAN = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # median of all detected patterns
    loops_peak_selected = []
    n_patterns = 0
    for fich in list_files:
        print(fich)
        chromo = fich
        #        chromo = "chr"+re.findall("chr""([0-9]+)", fich)[0]
        print(chromo)
        raw_test = np.loadtxt(fich)
        th_sum = np.median(raw_test.sum(axis=0)) - 2.0 * np.std(
            raw_test.sum(axis=0)
        )  # Removal of poor interacting bins
        indices_matrice = np.where(raw_test.sum(axis=0) > th_sum)
        indices_poor = np.where(raw_test.sum(axis=0) <= th_sum)
        matscn = scn_func(raw_test, th_sum)
        dd, matscn, n, outiers = despeckles(matscn, 10.0)

        dist = distance_law(matscn)
        x = np.arange(0, len(dist))
        y = dist
        y[np.isnan(y)] = 0.
        y_savgol = savgol_filter(y, window_length=17, polyorder=5)

        n1 = np.shape(matscn)[0]
        # Computation of genomic distance law matrice:
        MAT_DIST = np.zeros((n1, n1))
        for i in range(0, n1):
            for j in range(0, n1):
                MAT_DIST[i, j] = y_savgol[abs(j - i)]
        MAT_DETREND = matscn / MAT_DIST
        MAT_DETREND[np.isnan(MAT_DETREND)] = 1.0
        MAT_DETREND[MAT_DETREND < 0] = 1.0
        # refilling of empty bins with 1.0 (neutral):
        for i in indices_matrice[0]:
            MAT_DETREND[indices_poor[0], :] = np.ones(
                (len(indices_poor[0]), n1)
            )
            MAT_DETREND[:, indices_poor[0]] = np.ones(
                (n1, len(indices_poor[0]))
            )

        loops_peak_selected_chromo = []
        res2 = corrcoef2d(
            MAT_DETREND, p, centered_p=False
        )  # !!  Here the pattern match  !!
        res2[np.isnan(res2)] = 0.0
        n2 = np.shape(res2)[0]
        res_rescaled = np.zeros(np.shape(matscn))
        res_rescaled[
            np.ix_(
                range(int(area), n2 + int(area)),
                range(int(area), n2 + int(area)),
            )
        ] = res2
        VECT_VALUES = np.reshape(res_rescaled, (1, n1 * n1))
        VECT_VALUES = VECT_VALUES[0]
        thr = np.median(VECT_VALUES) + precision_value * np.std(VECT_VALUES)
        indices_max = np.where(res_rescaled > thr)
        indices_max = np.array(indices_max)
        res_rescaled = np.triu(res_rescaled)  # Centering:
        res_rescaled[(res_rescaled) < 0] = 0
        loops_peak = picker(res_rescaled, thr)  #   Recentring here !!

        if loops_peak != "NA":
            mask = np.array(abs(loops_peak[:, 0] - loops_peak[:, 1])) == 0
            loops_peak = loops_peak[mask, :]
            for l in loops_peak:
                if l[0] in indices_matrice[0] and l[1] in indices_matrice[0]:
                    p1 = int(l[0])
                    p2 = int(l[1])
                    if p1 > p2:
                        p22 = p2
                        p2 = p1
                        p1 = p22
                    if (
                        p1 - area >= 0
                        and p1 + area + 1 < n1
                        and p2 - area >= 0
                        and p2 + area + 1 < n1
                    ):
                        MAT_PANNEL = MAT_DETREND[
                            np.ix_(
                                range(p1 - area, p1 + area + 1),
                                range(p2 - area, p2 + area + 1),
                            )
                        ]
                        if (
                            len(MAT_PANNEL[MAT_PANNEL == 1.])
                            < ((area * 2 + 1) ** 2) * 20.0 / 100.
                        ):
                            n_patterns += 1
                            score = res_rescaled[l[0], l[1]]
                            loops_peak_selected.append(
                                [chromo, l[0], l[1], score]
                            )
                            MAT_SUM = MAT_SUM + MAT_PANNEL
                            MAT_LIST.append(MAT_PANNEL)
                            LIST_SIZES.append(abs(p2 - p1))
                        else:
                            loops_peak_selected.append(
                                [chromo, "NA", "NA", "NA"]
                            )
        else:
            loops_peak_selected.append([chromo, "NA", "NA", "NA"])

    # Computation of stats on the whole set - Agglomerated procedure :
    for i in range(0, area * 2 + 1):
        for j in range(0, area * 2 + 1):
            list_temp = []
            for el in range(1, len(MAT_LIST)):
                list_temp.append(MAT_LIST[el][i, j])
            MAT_MEDIAN[i, j] = np.median(list_temp)

    return loops_peak_selected, LIST_SIZES, MAT_LIST, MAT_MEDIAN


def convolve2d(signal, kernel, centered_p=True):
    """
    Convolution of a 2 diemensional signal with a kernel.
    INPUT:
    signal: a 2-dimensional numpy array Ms x Ns
    kernel: a 2-dimensional numpy array Mk x Nk
    centered_p: if False then return a matrix of size return a matrix of size (Ms-Mk+1) x (Ns-Nk+1)
                otherwise return a matrix of size Ms x Ns whith values located at center of kernel.
                (Default is True)
    OUTPUT:
    out: 2-dimensional numpy array corresponding of signal convolved by kernel.
         The size of out depends on whether cenetred_p is True or False
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
    """
    Pearson correlation coefficient between signal and sliding kernel.
    """
    kernel1 = np.ones(kernel.shape) / kernel.size
    mean_signal = convolve2d(signal, kernel1, centered_p)
    std_signal = np.sqrt(
        convolve2d(signal ** 2, kernel1, centered_p) - mean_signal ** 2
    )
    mean_kernel = np.mean(kernel)
    std_kernel = np.std(kernel)
    corrcoef = (
        convolve2d(signal, kernel / kernel.size, centered_p)
        - mean_signal * mean_kernel
    ) / (std_signal * std_kernel)
    return corrcoef


if __name__ == "__main__":

    signal = np.arange(10)
    for ii in range(9):
        signal = np.vstack((signal, np.arange(10)))
    kernel = (
        np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        + 0.0
    )
    signal2 = convolve2d(signal, kernel)

    cor = corrcoef2d(signal, kernel)
