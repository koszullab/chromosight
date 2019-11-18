#!/usr/bin/env python3
# coding: utf-8

"""Preprocessing utils
Operations to perform on Hi-C matrices before analyses
"""
import numpy as np
import sys
from scipy.signal import savgol_filter
import scipy.stats as ss
from scipy.sparse import dia_matrix, csr_matrix, coo_matrix, issparse
import scipy.ndimage as ndi


def normalize(B, good_bins=None, iterations=10):
    """
    Iterative normalisation of a Hi-C matrix (ICE procedure, 
    Imakaev et al, doi: 10.1038/nmeth.2148)

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
    r = r.tocsr()
    # Make a boolean mask from good bins
    good_mask = np.isin(range(r.shape[0]), good_bins)
    # Set all pixels in a nondetectable bin to 0
    r[~good_mask, :] = 0
    r[:, ~good_mask] = 0
    r = r.tocoo()
    # Update sparsity and get coordinates of nonzero pixels
    r.eliminate_zeros()
    nz_rows, nz_cols = r.nonzero()
    for _ in range(iterations):
        bin_sums = sum_mat_bins(r)
        # Divide sums by their mean to avoid instability
        bin_sums /= np.median(bin_sums)
        # ICE normalisation (Divide each valid pixel by the product of its row and column)
        r.data /= np.float64(bin_sums[nz_rows] * bin_sums[nz_cols])
    bin_sums = sum_mat_bins(r)
    # Scale to 1
    r.data = r.data * (1 / np.median(bin_sums))
    r.eliminate_zeros()

    return r


def diag_trim(mat, n):
    """
    Trim an upper triangle sparse matrix so that only the first n diagonals are kept.

    Parameters
    ----------

    mat : scipy.sparse.dia_matrix
        The sparse matrix to be trimmed
    n : int
        The number of diagonals from the center to keep (0-based).

    Returns
    -------
    scipy.sparse.dia_matrix :
        The diagonally trimmed upper triangle matrix with only the first n diagonal.
    """
    if not issparse(mat) or mat.format != "dia":
        raise ValueError("input type must be scipy.sparse.dia_matrix")
    # Create a new matrix from the diagonals below max dist (faster than removing them)
    keep_offsets = np.where((mat.offsets <= n) & (mat.offsets >= 0))[0]
    trimmed = dia_matrix(
        (mat.data[keep_offsets], mat.offsets[keep_offsets]), shape=mat.shape
    )

    return trimmed


def distance_law(matrix, detectable_bins=None, fun=np.nanmedian):
    """
    Computes genomic distance law by averaging over each diagonal in
    the upper triangle matrix.

    parameters
    ----------
    matrix: scipy.sparse.csr_matrix
        the input matrix to compute distance law from.
    detectable_bins : numpy.array of ints
        An array of detectable bins indices to consider when computing distance law.
    fun : callable
        A function to apply on each diagonal. Defaults to mean.

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
        # Find detectable which fall in diagonal
        detect_mask = np.zeros(n, dtype=bool)
        detect_mask[detectable_bins] = 1
        # Find bins which are detectable in the diagonal (intersect of hori and verti)
        detect_mask_h = detect_mask[: (n - diag)]
        detect_mask_v = detect_mask[n - (n - diag) :]
        detect_mask_diag = detect_mask_h & detect_mask_v
        dist[diag] = fun(matrix.diagonal(diag)[detect_mask_diag])
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
    # Compute median and MAD for each diagonal
    medians, stds = {}, {}
    for u in dist:
        medians[u] = np.median(dist[u])
        stds[u] = ss.median_absolute_deviation(dist[u], nan_policy="omit")

    # Loop over all nonzero pixels in the COO matrix and their coordinates
    for i, (row, col, val) in enumerate(
        zip(matrix.row, matrix.col, matrix.data)
    ):
        # Compute genomic distance of interactions in pixel
        dist = abs(row - col)
        # If pixel in input matrix is an outlier, set this pixel to median
        # of its diagonal in output matrix
        if val > medians[dist] + th2 * stds[dist]:
            A.data[i] = medians[dist]
    return A


def get_detectable_bins(mat, n_mads=3, inter=False):
    """
    Returns lists of detectable indices after excluding low interacting bin
    based on the distribution of pixel values in the matrix.

    Parameters
    ----------
    mat : scipy.sparse.coo_matrix
        A Hi-C matrix in tihe form of a 2D numpy array or coo matrix
    n_mads : int
        Number of median absolute deviation below the median required to
        consider bins non-detectable.
    inter : bool
        Whether the matrix is interchromosomal. Default is to consider the matrix
        is intrachromosomal (i.e. upper symmetric).

    Returns
    -------
    numpy array :
        tuple of 2 1D arrays containing indices of low interacting rows and
        columns, respectively.
    -------
    """
    matrix = mat.copy()
    mad = lambda x: ss.median_absolute_deviation(x, nan_policy="omit")
    if not inter:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("intrachromosomal matrices must be symmetric.")
        matrix.data = np.ones(matrix.data.shape)
        sum_bins = sum_mat_bins(matrix)
        sum_mad = mad(sum_bins)
        # Find poor interacting rows and columns
        sum_med = np.median(sum_bins)
        detect_threshold = max(1, sum_med - sum_mad * n_mads)
        # Removal of poor interacting rows and columns
        good_bins = np.where(sum_bins > detect_threshold)[0]
        good_bins = (good_bins, good_bins)
    else:
        sum_rows, sum_cols = matrix.sum(axis=0), matrix.sum(axis=1)
        mad_rows, mad_cols = mad(sum_rows), mad(sum_cols)
        med_rows, med_cols = np.median(sum_rows), np.median(sum_cols)
        detect_threshold_rows = max(1, med_rows - mad_rows * n_mads)
        detect_threshold_cols = max(1, med_cols - mad_cols * n_mads)
        good_rows = np.where(sum_rows > detect_threshold_rows)[0]
        good_cols = np.where(sum_cols > detect_threshold_cols)[0]
        good_bins = (good_rows, good_cols)
    return good_bins


def detrend(matrix, detectable_bins=None, fun=np.nanmedian):
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
    y = distance_law(matrix, detectable_bins)
    y[np.isnan(y)] = 0.0

    # Detrending by the distance law
    clean_mat = matrix.tocoo()
    # clean_mat.data /= y_savgol[abs(clean_mat.row - clean_mat.col)]
    clean_mat.data /= y[abs(clean_mat.row - clean_mat.col)]
    clean_mat = clean_mat.tocsr()
    # Set values in bad bins to 0
    miss_bin_mask = np.ones(matrix.shape[0], dtype=bool)
    miss_bin_mask[detectable_bins] = 0
    clean_mat[np.ix_(miss_bin_mask, miss_bin_mask)] = 0.0
    clean_mat.eliminate_zeros()

    return clean_mat


def ztransform(matrix):
    """
    Z transformation for Hi-C matrices.

    Parameters
    ----------
    matrix : numpy.array
        A 1-dimensional numpy array of nonzero values from a sparse matrix.

    Returns
    -------
    numpy.array :
        A 1-dimensional numpy array of the z-transformed nonzero values.
    """

    data = matrix.data
    mu = np.mean(data)
    sd = np.std(data)
    N = matrix.copy()
    N.data -= mu
    N.data /= sd

    return N


def signal_to_noise_threshold(matrix, detectable_bins, smooth=True):
    """
    Compute signal to noise ratio (SNR) at each diagonal of the matrix to determine
    the maximum scanning distance from the diagonal. The SNR is smoothed using
    the savgol filter.

    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix
        The Hi-C contact map in sparse format.
    detectable_bins : numpy.array
        Array containing indices of detectable bins.
    smooth : bool
        Whether the SNR values should be smoothed using the savgol filter.
    Returns
    -------
    int :
        The maximum distance from the diagonal at which the matrix should be scanned
    """
    # Using median and mad to reduce sensitivity to outlier
    dist_medians = distance_law(matrix, detectable_bins, np.median)
    dist_mads = distance_law(
        matrix,
        detectable_bins,
        lambda x: ss.median_absolute_deviation(x, nan_policy="omit"),
    )
    snr = dist_medians / dist_mads
    # Values below 1 are considered too noisy
    threshold_noise = 1.0
    snr[np.isnan(snr)] = 0.0
    try:
        if smooth:
            snr = savgol_filter(snr, window_length=17, polyorder=5)
        max_dist = min(np.where(snr < threshold_noise)[0])
    except ValueError:
        max_dist = matrix.shape[0]
    return max_dist


def sum_mat_bins(mat):
    """
    Compute the sum of matrices bins (i.e. rows or columns) using
    only the upper triangle, assuming symmetrical matrices.

    Parameters
    ----------
    mat : scipy.sparse.coo_matrix
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


def subsample_contacts(M, n_contacts):
    """Bootstrap sampling of contacts in a sparse Hi-C map.

    Parameters
    ----------
    M : scipy.sparse.coo_matrix
        The input Hi-C contact map in sparse format.
    n_contacts : int 
        The number of contacts to sample.

    Returns
    -------
    scipy.sparse.coo_matrix
        A new matrix with a fraction of the original contacts.
    """

    S = M.data.copy()
    # Match cell idx to cumulative number of contacts
    cum_counts = np.cumsum(S)
    # Total number of contacts to sample
    tot_contacts = int(cum_counts[-1])

    # Sample desired number of contacts from the range(0, n_contacts) array
    sampled_contacts = np.random.choice(
        int(tot_contacts), size=(n_contacts), replace=False
    )

    # Get indices of sampled contacts in the cum_counts array
    idx = np.searchsorted(cum_counts, sampled_contacts, side="right")

    # Bin those indices to the same dimensions as matrix data to get counts
    sampled_counts = np.bincount(idx, minlength=S.shape[0])

    # Get nonzero values to build new sparse matrix
    nnz_mask = sampled_counts > 0
    sampled_counts = sampled_counts[nnz_mask].astype(np.float64)
    sampled_rows = M.row[nnz_mask]
    sampled_cols = M.col[nnz_mask]

    return coo_matrix(
        (sampled_counts, (sampled_rows, sampled_cols)),
        shape=(M.shape[0], M.shape[1]),
    )


def resize_kernel(kernel, kernel_res, signal_res, min_size=5, max_size=101):
    """
    Resize a kernel matrix based on the resolution at which it was defined and
    the signal resolution. E.g. if a kernel matrix was generated for 10kb and
    the input signal is 20kb, kernel size will be divided by two. If the kernel
    is enlarged, pixels are interpolated with a spline of degree 1.

    Parameters
    ----------
    kernel : numpy.array
        Kernel matrix.
    kernel_res : int
        Resolution for which the kernel was designed.
    signal_res : int
        Resolution of the signal matrix in basepair per matrix bin.
    min_size : int
        Lower bound, in number of rows/column allowed when resizing the kernel.
    max_size : int
        Upper bound, in number of rows/column allowed when resizing the kernel.
    
    Returns
    -------
    resized_kernel : numpy.array
        The resized input kernel.
    """
    km, kn = kernel.shape

    if km != kn:
        ValueError("kernel must be square.")
    if not (km % 2) or not (kn % 2):
        ValueError("kernel size must be odd.")

    # Define by how many times kernel must be enlarged for its pixels to match
    # the signal's pixels
    resize_factor = kernel_res / signal_res
    if km * resize_factor > max_size:
        resize_factor = max_size / km
    elif km * resize_factor < min_size:
        resize_factor = min_size / km

    resized_kernel = ndi.zoom(kernel, resize_factor, order=1)

    # TODO: Adjust resize factor to ensure odd dimensions before zooming
    # instead of trimming afterwards
    if not resized_kernel.shape[0] % 2:
        # Compute the factor required to yield a dimension smaller by one
        adj_resize_factor = (resized_kernel.shape[0] - 1) / km
        sys.stderr.write(
            f"Adjusting resize factor from {resize_factor} to {adj_resize_factor}.\n"
        )
        resized_kernel = ndi.zoom(kernel, adj_resize_factor, order=1)

    return resized_kernel
