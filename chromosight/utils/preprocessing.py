#!/usr/bin/env python3
# coding: utf-8

"""Preprocessing utils
Operations to perform on Hi-C matrices before analyses
"""
import sys
import numpy as np
import numpy.linalg as la
import scipy.stats as ss
import scipy.sparse as sp
import scipy.ndimage as ndi
from sklearn.isotonic import IsotonicRegression


def erase_missing(signal, valid_rows, valid_cols, sym_upper=True):
    """
    Given an input sparse matrix, set missing (invalid) bins to 0.

    Parameters
    ----------
    signal : scipy.sparse.csr_matrix of floats
        Input signal on which to erase values.
    valid_rows : numpy.array of ints
        Indices of rows considered valid (not missing).
    valid_cols : numpy.array of ints
        Indices of columns considered valid (not missing).
    sym_upper : bool
        Define if the input signal is upper symmetric.

    Returns
    -------
    scipy.sparse.csr_matrix
        The input signal with all values in missing bins set to 0
    """
    if sym_upper and sp.issparse(signal):
        if np.any(valid_rows != valid_cols):
            raise ValueError(
                "Valid rows and columns must be identical with sym_upper=True"
            )
        if signal.shape[0] != signal.shape[1]:
            raise ValueError(
                "Input matrix must be square when using sym_upper=True"
            )
        # Make a boolean mask from good bins
        good_mask = np.isin(range(signal.shape[0]), valid_rows)
        # Set all pixels in a nondetectable bin to 0
        # For faster masking of bins, mask bins using dot product with an
        # identify matrix where bad bins have been masked on the diagonal
        # E.g. if removing the second bin (row and column):
        # 1 0 0     9 6 5     1 0 0     9 0 5
        # 0 0 0  X  6 8 7  X  0 0 0  =  0 0 0
        # 0 0 1     6 7 8     0 0 1     6 0 8
        mask_mat = sp.eye(signal.shape[0])
        mask_mat.data[0][~good_mask] = 0
        erased = mask_mat.dot(signal).dot(mask_mat)
    else:
        # Get a boolean array of missing (1) and valid (0) rows
        missing_rows = np.ones(signal.shape[0])
        missing_cols = np.ones(signal.shape[1])
        missing_rows[valid_rows] = 0
        missing_cols[valid_cols] = 0
        missing_rows = np.where(missing_rows)[0]
        missing_cols = np.where(missing_cols)[0]
        erased = signal.copy()
        erased[missing_rows, :] = 0
        erased[:, missing_cols] = 0

    return erased


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
    scipy.sparse.coo_matrix :
        The SCN normalised Hi-C matrix
    """
    if good_bins is None:
        good_bins = np.arange(B.shape[0])
        r = B.copy()
    else:
        # Enforce removal of values in missing bins
        r = erase_missing(B, good_bins, good_bins, sym_upper=True)
    r = r.tocoo()
    # Update sparsity and get coordinates of nonzero pixels
    r.eliminate_zeros()
    nz_rows, nz_cols = r.nonzero()
    for _ in range(iterations):
        bin_sums = sum_mat_bins(r)
        # Divide sums by their mean to avoid instability
        bin_sums /= np.median(bin_sums)
        # ICE normalisation (Divide each valid pixel by the product
        # of its row and column)
        r.data /= np.float64(bin_sums[nz_rows] * bin_sums[nz_cols])
    bin_sums = sum_mat_bins(r)
    # Scale to 1
    r.data = r.data * (1 / np.median(bin_sums))
    r.eliminate_zeros()
    return r


def set_mat_diag(mat, diag=0, val=0):
    """
    Set the nth diagonal of a symmetric 2D numpy array to a fixed value.
    Operates in place.

    Parameters
    ----------
    mat : numpy.array
        Symmetric 2D array of floats.
    diag : int
        0-based index of the diagonal to modify. Use negative values for the
        lower half.
    val : float
        Value to use for filling the diagonal
    """
    m = mat.shape[0]
    step = m + 1
    start = diag
    end = m ** 2 - diag * m
    mat.flat[start:end:step] = val


def diag_trim(mat, n):
    """
    Trim an upper triangle sparse matrix so that only the first n diagonals
    are kept.

    Parameters
    ----------

    mat : scipy.sparse.csr_matrix or numpy.array
        The sparse matrix to be trimmed
    n : int
        The number of diagonals from the center to keep (0-based).

    Returns
    -------
    scipy.sparse.dia_matrix or numpy.array:
        The diagonally trimmed upper triangle matrix with only the first
        n diagonal.
    """
    if not sp.issparse(mat):
        trimmed = mat.copy()
        n_diags = trimmed.shape[0]
        for diag in range(n, n_diags):
            set_mat_diag(trimmed, diag, 0)
        return trimmed

    if mat.format != "csr":
        raise ValueError("input type must be scipy.sparse.csr_matrix")
    # Trim diagonals by removing all elements further than n in the
    # upper triangle
    trimmed = sp.tril(mat, n, format="csr")
    trimmed = sp.triu(trimmed, format="csr")

    return trimmed


def distance_law(
    matrix, detectable_bins=None, max_dist=None, smooth=True, fun=np.nanmean
):
    """
    Computes genomic distance law by averaging over each diagonal in
    the upper triangle matrix.

    parameters
    ----------
    matrix: scipy.sparse.csr_matrix
        the input matrix to compute distance law from.
    detectable_bins : numpy.array of ints
        An array of detectable bins indices to consider when computing
        distance law.
    max_dist : int
        Maximum distance from diagonal, in number of bins in which to compute
        distance law
    smooth : bool
        Whether to use isotonic regression to smooth the distance law.
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
    mat_n = matrix.shape[0]
    if max_dist is None:
        max_dist = mat_n
    n_diags = min(mat_n, max_dist + 1)
    dist = np.zeros(mat_n)
    if detectable_bins is None:
        detectable_bins = np.array(range(mat_n))

    for diag in range(n_diags):
        # Find detectable which fall in diagonal
        detect_mask = np.zeros(mat_n, dtype=bool)
        detect_mask[detectable_bins] = 1
        # Find bins which are detectable in the diagonal (intersect of
        # hori and verti)
        detect_mask_h = detect_mask[: (mat_n - diag)]
        detect_mask_v = detect_mask[mat_n - (mat_n - diag) :]
        detect_mask_diag = detect_mask_h & detect_mask_v
        detect_diag = matrix.diagonal(diag)[detect_mask_diag]
        dist[diag] = fun(detect_diag[detect_diag > 0])
    # Smooth the curve using isotonic regression: Find closest approximation
    # with the condition that point n+1 cannot be higher than point n.
    # (i.e. contacts can only decrease when increasing distance)
    if smooth and mat_n > 2:
        ir = IsotonicRegression(increasing=False)
        dist[~np.isfinite(dist)] = 0
        dist = ir.fit_transform(range(len(dist)), dist)

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
        Whether the matrix is interchromosomal. Default is to consider the
        matrix is intrachromosomal (i.e. upper symmetric).

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
        # Replace nonzero pixels by ones to work on prop. of nonzero pixels
        matrix.data = np.ones(matrix.data.shape)
        # Compute number of nonzero values in each bin
        sum_bins = sum_mat_bins(matrix)
        # Compute variation in the number of nonzero pixels
        sum_mad = mad(sum_bins)
        # Find poor interacting rows and columns
        sum_med = np.median(sum_bins)
        detect_threshold = max(1, sum_med - sum_mad * n_mads)

        # Removal of poor interacting rows and columns
        good_bins = np.where(sum_bins > detect_threshold)[0]
        good_bins = (good_bins, good_bins)
    else:
        # Adapted for asymetric matrices (need to compute rows and columns)
        sum_rows, sum_cols = matrix.sum(axis=1).A1, matrix.sum(axis=0).A1
        mad_rows, mad_cols = mad(sum_rows), mad(sum_cols)
        med_rows, med_cols = np.median(sum_rows), np.median(sum_cols)
        detect_threshold_rows = max(1, med_rows - mad_rows * n_mads)
        detect_threshold_cols = max(1, med_cols - mad_cols * n_mads)
        good_rows = np.where(sum_rows > detect_threshold_rows)[0]
        good_cols = np.where(sum_cols > detect_threshold_cols)[0]
        good_bins = (good_rows, good_cols)
    return good_bins


def detrend(
    matrix,
    detectable_bins=None,
    max_dist=None,
    smooth=False,
    fun=np.nanmedian,
    max_val=10,
):
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
    max_dist : int
        Maximum number of bins from the diagonal at which to compute trend.
    smooth : bool
        Whether to use isotonic regression to smooth the trend.
    fun : callable
        Function to use on each diagonal to compute the trend.
    max_val : float
        Maximum value in the detrended matrix.

    Returns
    -------
    numpy.ndarray :
        The detrended Hi-C matrix.
    """
    matrix = matrix.tocsr()
    y = distance_law(
        matrix,
        detectable_bins=detectable_bins,
        max_dist=max_dist,
        smooth=smooth,
        fun=fun,
    )
    y[np.isnan(y)] = 0.0

    # Detrending by the distance law
    clean_mat = matrix.tocoo()
    # clean_mat.data /= y_savgol[abs(clean_mat.row - clean_mat.col)]
    try:
        clean_mat.data /= y[abs(clean_mat.row - clean_mat.col)]
    # If no nonzero value in matrix, do nothing
    except TypeError:
        pass
    clean_mat = clean_mat.tocsr()
    clean_mat[clean_mat >= max_val] = max_val
    return clean_mat


def ztransform(matrix):
    """
    Z transformation for Hi-C matrices.

    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix
        A Hi-C matrix in sparse format.

    Returns
    -------
    scipy.sparse.coo_matrix:
        The detrended Hi-C matrix
    """

    mat = matrix.copy()
    mu = np.mean(mat.data)
    sd = np.std(mat.data)
    mat.data -= mu
    mat.data /= sd

    return mat


def signal_to_noise_threshold(matrix, detectable_bins):
    """
    Compute signal to noise ratio (SNR) at each diagonal of the matrix to
    determine the maximum scanning distance from the diagonal.

    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix
        The Hi-C contact map in sparse format.
    detectable_bins : numpy.array
        Array containing indices of detectable bins.
    Returns
    -------
    int :
        The maximum distance from the diagonal at which the matrix should be
        scanned
    """
    # Using median and mad to reduce sensitivity to outlier
    dist_means = distance_law(
        matrix, detectable_bins=detectable_bins, fun=np.nanmean
    )
    dist_stds = distance_law(
        matrix, detectable_bins=detectable_bins, fun=np.nanstd
    )
    snr = dist_means / dist_stds
    # Values below 1 are considered too noisy
    threshold_noise = 1.0
    snr[np.isnan(snr)] = 0.0
    try:
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

    return sp.coo_matrix(
        (sampled_counts, (sampled_rows, sampled_cols)),
        shape=(M.shape[0], M.shape[1]),
    )


def frame_missing_mask(mask, kernel_shape, sym_upper=False, max_dist=None):
    """
    Adds a frame around input mask, given a kernel. The goal of this
    frame is define margins around the matrix where the kernel will not perform
    convolution (denoted by 1). If the matrix is upper symmetric, a margin of
    half the kernel's width is added below the diagonal and a maximum distance
    from the diagonal above which margins need not be drawn can be considered.
    Otherwise Margins are simply added on all 4 sides of the matrix.

    signal    kernel   _________
    ______   ____      |#######|
    |     |  |   | ==> |#     #|
    |     |  |___|     |#     #|
    |     |            |#     #|
    |_____|            |#     #|
                       |#######|
                       --------

    Parameters
    ----------
    mask : scipy.sparse.csr_matrix of bool
        The mask around which to add margins.
    kernels_shape : tuple of ints
        The number of rows and kernel in the input kernel. Margins will be half
        these values.
    sym_upper : bool
        Whether the signal is a symmetric upper triangle matrix. If so, values
        on a margin below the diagonal will be masked.
    max_dist : int or None
        Number of diagonals to keep

    Returns
    -------
    framed_mask : scipy.sparse.csr_matrix of bool
        The input mask with a padding of True around the edges. If sym_upper
        is True, a padding is also added below the diagonal.
    """
    if mask.dtype != bool:
        raise ValueError("Mask must contain boolean values")
    if not sp.issparse(mask):
        raise ValueError("Mask must be a sparse matrix")

    framed_mask = mask.copy()
    ms, ns = mask.shape
    mk, nk = kernel_shape
    if sym_upper and (max_dist is not None):
        # Remove diagonals further than scan distance in the input mask
        framed_mask = diag_trim(framed_mask, max_dist + max(nk, mk)).tocsr()
        max_m = max_dist + mk
        max_n = max_dist + nk
    else:
        max_m, max_n = ms, ns
    # Up and down margins initialized with zeros and filled as needed
    margin_1 = sp.csr_matrix((mk - 1, ns), dtype=bool)
    margin_2 = sp.csr_matrix((mk - 1, ns), dtype=bool)
    if sym_upper and (max_dist is not None):
        # Margin 1 (top) is in upper triangle -> fill missing up to scan dist
        margin_1[:, :max_n] = 1
    else:
        margin_1[:, :] = 1
        margin_2[:, :] = 1
    framed_mask = sp.vstack([margin_1, framed_mask, margin_2], format="csr")

    # Left and right
    margin_1 = sp.csr_matrix((ms + 2 * (mk - 1), nk - 1), dtype=bool)
    margin_2 = sp.csr_matrix((ms + 2 * (mk - 1), nk - 1), dtype=bool)

    if sym_upper and (max_dist is not None):
        # Margin 2 (right) is in upper triangle-> fill missing up to scan dist
        margin_2[-(max_m + 1) :, :] = 1
        # Fill only the start of left margin for the top-left corner
        margin_1[: mk - 1, :] = 1
    else:
        margin_1[:, :] = 1
        margin_2[:, :] = 1
    framed_mask = sp.hstack([margin_1, framed_mask, margin_2], format="csr")

    if sym_upper:
        # LIL format is much faster when changing sparsity
        framed_mask = framed_mask.tolil()
        # Add margin below diagonal
        big_k = max(nk, mk)
        dia_margin = np.ones(big_k // 2 - 1)
        dia_offsets = np.arange(-1, -big_k // 2 + 1, -1)
        framed_mask += sp.diags(
            dia_margin,
            dia_offsets,
            shape=framed_mask.shape,
            format="lil",
            dtype=bool,
        )
        framed_mask = framed_mask.tocsr()
    return framed_mask


def check_missing_mask(signal, mask):
    """
    Ensure all elements defined as missing by the mask are set to zero in the
    signal. If this is not the case, raises an error.

    Parameters
    ----------
    signal : numpy.array of floats or scipy.sparse.csr_matrix of floats
        The signal to be checked.
    mask : numpy.array of bools or scipy.sparse.csr_matrix of bools
        The mask defining missing values as True and valid values as False.
    """

    if sp.issparse(mask):
        # Check if there are nonzero values in the signal reported as missing
        # by the mask
        missing_with_signal = np.nonzero(
            abs(signal[mask.nonzero()[0], mask.nonzero()[1]]) > 0
        )[0]
        if len(missing_with_signal) > 0:
            raise ValueError(
                "There are",
                len(missing_with_signal),
                "non-zero elements reported as missing.",
            )
    else:
        if np.sum(abs(signal[mask > 0])) > 1e-10:
            raise ValueError(
                "There are",
                str(np.sum(abs(signal[mask > 0]))),
                "non-zero elements reported as missing.",
            )


def make_missing_mask(
    shape, valid_rows, valid_cols, max_dist=None, sym_upper=False
):
    """
    Given lists of valid rows and columns, generate a sparse matrix mask with
    missing pixels denoted as 1 and valid pixels as 0. If a max_dist is
    provided, upper symmetric matrices will only be flagged up to max_dist
    pixels from the diagonal.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to generate.
    valid_rows : numpy.array of ints
        Array with the indices of valid rows that should be set to 0 in the
        mask.
    valid_cols : numpy.array of ints
        Array with the indices of valid rows that should be set to 0 in the
        mask.
    max_dist : int or None
        The maximum diagonal distance at which masking should take place.
    sym_upper : bool
        Whether the matrix is symmetric upper. If so, max_dist is ignored

    Returns
    -------
    scipy.sparse.csr_matrix of bool
        The mask containing False values where pixels are valid and True valid
        where pixels are missing
    """
    # Error if the matrix upper symmetric but shape is rectangle or missing
    # rows and cols are different
    sm, sn = shape
    if sym_upper and (sm != sn or len(valid_rows) != len(valid_cols)):
        raise ValueError("Rectangular matrices cannot be upper symmetric")

    # Get a boolean array of missing (1) and valid (0) rows
    missing_rows = np.ones(sm, dtype=bool)
    missing_rows[valid_rows] = False
    missing_rows = np.where(missing_rows)[0]
    # When matrix is sym., rows and cols are synonym, no need to compute 2x
    if sym_upper:
        missing_cols = missing_rows
    else:
        missing_cols = np.ones(sn, dtype=bool)
        missing_cols[valid_cols] = False
        missing_cols = np.where(missing_cols)[0]

    # If upper sym., fill only upper diag up to max_dist.
    # E. g. with bins 1 and 3 missing
    # and a max_dist of 1:
    # 0 1 0 0 0
    # 0 1 1 0 0
    # 0 0 0 1 0
    # 0 0 0 1 1
    # 0 0 0 0 0
    # For each missing bin, mask is apply 1 pixel upwards and 1 to the right
    # to fill only the upper triangle up to max_dist
    if sym_upper:
        # If no max dist has been specified, fill the whole upper triangle
        if max_dist is None:
            max_dist = min(shape)
        # Generate matrix of distance shifts by row.
        # Shape is len(missing_rows) x (max_dist + 1)
        # e.g.: 2 missing rows and max dist of 1
        # 0 0
        # 1 1
        row_shifts = np.tile(
            np.array(range(max_dist + 1)), (len(missing_rows), 1)
        ).T
        # Compute row positions upwards to diagonal by subtracting missing rows
        # to the shifts. Following the previous example, if missing rows are
        # bins 1 and 3:
        #  1 3
        #  0 2
        rows_before = (missing_rows - row_shifts).flatten("F")
        # looking at pixels up from the bins, cols remain the same:
        # 1 3
        # 1 3
        cols_before = np.repeat(missing_rows, max_dist+1)
        # Compute col position to the right until diagonal by adding the shift
        # Note: upper symmetric, so row_shifts = col_shift_
        # 1 3
        # 2 4
        cols_after = (missing_cols + row_shifts).flatten("F")
        # This time, rows remain constant since we are computing positions to
        # the right
        rows_after = np.repeat(missing_cols, max_dist+1)
        # Combine positions to the right and upwards
        rows = np.concatenate([rows_before, rows_after])
        cols = np.concatenate([cols_before, cols_after])
        data = np.ones(rows.shape, dtype=bool)
        # Remove entries where rows or cols are negative or larger than shape
        valid = (cols < sm) & (cols >= 0) & (rows < sm) & (rows >= 0)
        # Build mask mat with miss bins up to max scan dist in upper triangle
        mask = sp.coo_matrix(
            (data[valid], (rows[valid], cols[valid])), shape=shape, dtype=bool
        ).tocsr()
    else:
        mask = sp.csr_matrix(shape, dtype=bool)
        mask[missing_rows, :] = 1
        mask[:, missing_cols] = 1

    return mask


def zero_pad_sparse(mat, margin_h, margin_v, fmt="coo"):
    """
    Adds margin of zeros around an input sparse matrix.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The matrix to be padded.
    margin_h : int
        The width of the horizontal margin to add on the left and right of the
        matrix.
    margin_v : int
        The width of the vertical margin to add on the top and bottom of the
        matrix.
    fmt : string
        The desired scipy sparse format of the output matrix

    Returns
    -------
    scipy.sparse.csr_matrix :
        The padded matrix of dimensions (m + 2 * margin_h, n + 2 * margin_v).

    Examples
    --------
    >>> m = sp.csr_matrix(np.array([[1, 2], [10, 20]]))
    >>> zero_pad_sparse(m, 2, 1).toarray()
    array([[ 0,  0,  0,  0,  0,  0],
           [ 0,  0,  1,  2,  0,  0],
           [ 0,  0, 10, 20,  0,  0],
           [ 0,  0,  0,  0,  0,  0]])
    """

    sm, sn = mat.shape
    padded_mat = mat.copy()
    # Up and down margins initialized with zeros and filled as needed
    margin_h_0 = sp.csr_matrix((sm, margin_h), dtype=mat.dtype)
    margin_v_0 = sp.csr_matrix((margin_v, sn + 2 * margin_h), dtype=mat.dtype)
    padded_mat = sp.hstack([margin_h_0, padded_mat, margin_h_0], format="csr")
    padded_mat = sp.vstack([margin_v_0, padded_mat, margin_v_0], format="csr")

    return padded_mat


def crop_kernel(kernel, target_size):
    """
    Crop a kernel matrix to target size horizontally and vertically.
    If the target size is even, the target size is adjusted to the
    next integer up.

    Parameters
    ----------
    kernel : numpy.array of floats
        Image to crop.
    target_size : tuple of ints
        Tuple defining the target shape of the kernel, takes the
        form (rows, cols) where rows and cols are odd numbers.

    Returns
    -------
    cropped : numpy.array of floats
        New image no larger than target dimensions
    """
    # Use list for mutability
    target = [d for d in target_size]
    adjusted = False
    for dim in range(len(target)):
        if not target[dim] % 2:
            target[dim] += 1
            adjusted = True
    if adjusted:
        sys.stderr.write(
            "WARNING: Cropped kernel size adjusted to "
            f"{target[0]}x{target[1]} to keep odd dimensions.\n"
        )

    source_m, source_n = kernel.shape
    target_m, target_n = target
    # Define horizontal and vertical margins to trim
    if source_m > target_m:
        margin_rows = (source_m - target_m) // 2
    else:
        margin_rows = 0
    if source_n > target_n:
        margin_cols = (source_n - target_n) // 2
    else:
        margin_cols = 0

    cropped = kernel[
        margin_rows : (source_m - margin_rows),
        margin_cols : (source_n - margin_cols),
    ]

    return cropped


def resize_kernel(
    kernel,
    kernel_res=None,
    signal_res=None,
    factor=None,
    min_size=7,
    max_size=101,
):
    """
    Resize a kernel matrix based on the resolution at which it was defined and
    the signal resolution. E.g. if a kernel matrix was generated for 10kb and
    the input signal is 20kb, kernel size will be divided by two. If the kernel
    is enlarged, pixels are interpolated with a spline of degree 1.
    Alternatively, a resize factor can be provided. In the example above, the
    factor would be 0.5.

    Parameters
    ----------
    kernel : numpy.array
        Kernel matrix.
    kernel_res : int
        Resolution for which the kernel was designed. Mutually exclusive with
        factor.
    signal_res : int
        Resolution of the signal matrix in basepair per matrix bin. Mutually
        exclusive with factor.
    factor : float
        Resize factor. Can be provided as an alternative to kernel_res and
        signal_res. Values above 1 will enlarge the kernel, values below 1 will
        shrink it.
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

    if factor is not None:
        if kernel_res is not None or signal_res is not None:
            raise ValueError(
                "factor is mutually exclusive with resolution "
                "parameters (kernel_res and signal_res)."
            )
        resize_factor = factor
    else:
        if kernel_res is None or signal_res is None:
            raise ValueError(
                "You must provide either a resize factor or the signal and "
                "kernel resolutions."
            )
        # Define by how many times kernel must be enlarged for its pixels to
        # match the signal's pixels
        resize_factor = kernel_res / signal_res
    if km * resize_factor > max_size:
        resize_factor = max_size / km
    elif km * resize_factor < min_size:
        resize_factor = min_size / km

    resized_kernel = ndi.zoom(kernel, resize_factor, order=1)

    if not resized_kernel.shape[0] % 2:
        # Compute the factor required to yield a dimension smaller by one
        adj_resize_factor = (resized_kernel.shape[0] - 1) / km
        sys.stderr.write(
            f"Adjusting resize factor from {resize_factor} to {adj_resize_factor}.\n"
        )
        resized_kernel = ndi.zoom(kernel, adj_resize_factor, order=1)

    return resized_kernel


def factorise_kernel(kernel, prop_info=0.999):
    """
    Performs truncated SVD on an input kernel, returning the singular vectors
    necessary to retain a given proportion of information contained in the
    kernel.

    Parameters
    ----------
    kernel : numpy.array of floats
        The input 2D kernel to factorise.
    prop_info : float
        Proportion of information to retain.

    Returns
    -------
    tuple of numpy.arrays
        A tuple containing the truncated left and right singular matrices,
        where each singular vector has been multiplied by the square root of
        their respective singular values.
    """
    u, sigma, v = la.svd(kernel)
    total_info = np.sum(sigma ** 2)
    # Compute min. number of singular vectors to retain enough info
    keep_k = np.where(np.cumsum(sigma ** 2) > prop_info * total_info)[0][0] + 1
    if keep_k > np.floor(min(kernel.shape) / 2):
        sys.stderr.write(
            f"Warning: Kernel factorisation required {keep_k} singular,"
            "vectors this may result in slow operations.\n",
        )
    # Truncate singular matrix to the keep only required vectors
    u = u[:, :keep_k]
    v = v[:keep_k, :]
    # Multiply each singular vector by the sqrt of its singular value
    for i in range(keep_k):
        u[:, i] *= np.sqrt(sigma[i])
        v[i, :] *= np.sqrt(sigma[i])

    return (u, v)
