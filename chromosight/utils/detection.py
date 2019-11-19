from __future__ import absolute_import
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss
import warnings
import chromosight.utils.preprocessing as preproc

warnings.filterwarnings("ignore")


def numba_jit():
    try:
        import numba
        import numba.autojit as jit
    except ImportError:
        warnings.warn(
            "Numba was not detected on this system, jit will not be enabled",
            ImportWarning,
        )
        jit = lambda u: u
    return jit


@numba_jit()
def validate_patterns(
    coords,
    matrix,
    conv_mat,
    detectable_bins,
    kernel_matrix,
    max_undetected_perc,
):
    """
    Filters detected patterns to remove those in noisy regions or too close to
    matrix boundaries. Also returns the surrounding window of Hi-C contacts around
    each detected pattern.

    Parameters
    ----------
    coords : numpy.array of int
        Coordinates of all detected patterns in the sub matrix. One pattern per
        row, the first column is the row number, second column is the col number.
    matrix : scipy.sparse.csr_matrix
        Hi-C contact map of the sub matrix.
    conv_mat : scipy.sparse.csr_matrix
        Convolution product of the kernel with the Hi-C sub matrix.
    detectable_bins : list of numpy.array
        List of two 1D numpy arrays of ints representing ids of detectable rows
        and columns, respectively.
    kernel_matrix : numpy.array of float
        The kernel that was used for pattern detection on the Hi-C matrix.
    max_undetected_perc : float
        Proportion of undetectable pixels allowed in a pattern window to consider
        it valid.

    Returns
    -------
    filtered_coords : pandas.DataFrame
        Table of coordinates that passed the filters. The dataframe has 3:
        columns: bin1 (rows), bin2 (col) and score (the correlation coefficient).
    filtered_windows : numpy.array
        3D numpy array of signal windows around detected patterns. Each window
        spans axes 1 and 2, and they are stacked along axis 0.
    """
    matrix = matrix.tocsr()
    # Pre-compute height, width and half (radius)
    win_h, win_w = kernel_matrix.shape
    half_h, half_w = win_h // 2 + 1, win_w // 2 + 1
    # Store coords to drop
    blacklist = []
    detectable_rows = set(detectable_bins[0])
    detectable_cols = set(detectable_bins[1])

    # Copy coords object and append column for scores
    validated_coords = pd.DataFrame(
        {
            "bin1": coords[:, 0],
            "bin2": coords[:, 1],
            "score": np.zeros(coords.shape[0]),
        }
    )
    # validated_coords = np.append(coords, np.zeros((coords.shape[0], 1)), 1)
    # Initialize structure to store pattern windows

    pattern_windows = np.zeros(
        (coords.shape[0], win_h, win_w)
    )  # list containing all pannel of detected patterns
    for i, l in enumerate(coords):
        p1 = int(l[0])
        p2 = int(l[1])
        if p1 > p2:
            p1, p2 = p2, p1
        # Check for out of bounds errors
        if (
            p1 - half_h >= 0
            and p1 + half_h + 1 < matrix.shape[0]
            and p2 - half_w >= 0
            and p2 + half_w + 1 < matrix.shape[1]
        ):
            # Get bin ids to use in window
            win_rows, win_cols = (
                range(p1 - half_h + 1, p1 + half_h),
                range(p2 - half_w + 1, p2 + half_w),
            )
            # Subset window from chrom matrix
            pattern_window = matrix[np.ix_(win_rows, win_cols)].todense()

            n_rows, n_cols = pattern_window.shape
            tot_pixels = n_rows * n_cols
            # Compute number of missing rows and cols
            n_bad_rows = n_rows - len(
                set(win_rows).intersection(detectable_rows)
            )
            n_bad_cols = n_cols - len(
                set(win_cols).intersection(detectable_cols)
            )

            # Number of undetected pixels is "bad rows area" + "bad cols area" - "bad rows x bad cols intersection"
            tot_undetected_pixels = (
                n_bad_rows * n_cols
                + n_bad_cols * n_rows
                - n_bad_rows * n_bad_cols
            )
            # Number of uncovered pixels
            tot_zero_pixels = len(pattern_window[pattern_window == 0].A1)
            tot_missing_pixels = max(tot_undetected_pixels, tot_zero_pixels)
            # The pattern should not contain more missing pixels that the max
            # value defined in kernel config. This includes both pixels from
            # undetectable bins and zero valued pixels in detectable bins.

            if tot_missing_pixels / tot_pixels < max_undetected_perc / 100.0:
                validated_coords.score[i] = conv_mat[l[0], l[1]]
                pattern_windows[i, :, :] = pattern_window
            else:
                # Current pattern will be dropped due to undetectable bins
                blacklist.append(i)
        else:
            # Current pattern will be dropped due to out of bound error
            blacklist.append(i)
    # Drop patterns that did not pass filters
    blacklist = np.array(blacklist)
    blacklist_mask = np.zeros(coords.shape[0], dtype=bool)
    if len(blacklist):
        blacklist_mask[blacklist] = True
    filtered_coords = validated_coords.loc[~blacklist_mask, :]
    filtered_windows = pattern_windows[~blacklist_mask, :, :]

    # from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(filtered_windows.shape[2], 1)
    # for i, axi in enumerate(ax.flatten()):
    #    axi.imshow(filtered_windows[:, :, i])
    # plt.show()

    return filtered_coords, filtered_windows


@numba_jit()
def pileup_patterns(pattern_windows):
    """Generate a pileup from an input list of pattern coords and a Hi-C matrix"""
    return np.apply_along_axis(np.median, 0, pattern_windows)


@numba_jit()
def pattern_detector(contact_map, kernel_config, kernel_matrix, area=3):
    """Pattern detector

    Detect patterns by iterated kernel matching, and extract windows around the
    detected patterns.

    Parameters
    ----------
    contact_map : ContactMap object
        An object containing an inter- or intra-chromosomal Hi-C contact map
        and additional metadata.
    kernel_config : dict
        The kernel configuration, as documented in
        chromosight.utils.io.load_kernel_config
    kernel_matrix : numpy.array
        The kernel matrix to use for convolution as a 2D numpy array
    area : int, optional
        The window size of the pileup pattern. The final windows will have
        a width of 2 * area + 1. Default is 8.
    Returns
    -------
    filtered_chrom_patterns : numpy.array
        A 2D array of detected patterns with 3 columns: x, y, score.
    chrom_pattern_windows : numpy array
        A 3D array containing the pile of windows around detected patterns.
    """

    # Do not attempt pattern detection unless matrix is larger than the kernel
    if kernel_matrix.shape[0] >= contact_map.matrix.shape[0]:
        return None, None

    # Pattern matching operate here
    mat_conv = corrcoef2d(
        contact_map.matrix, kernel_matrix, kernel_config["max_dist"]
    )
    mat_conv = mat_conv.tocoo()
    # Clean potential missing values
    mat_conv.data[np.isnan(mat_conv.data)] = 0
    # Only keep corrcoeff in scannable range
    mat_conv = preproc.diag_trim(mat_conv.todia(), contact_map.max_dist)
    mat_conv = mat_conv.tocoo()
    mat_conv.eliminate_zeros()

    # Find foci of highly correlated pixels
    chrom_pattern_coords = picker(mat_conv, kernel_config["precision"])
    if chrom_pattern_coords is None:
        return None, None
    filtered_chrom_patterns, chrom_pattern_windows = validate_patterns(
        chrom_pattern_coords,
        contact_map.matrix,
        mat_conv.tocsr(),
        contact_map.detectable_bins,
        kernel_matrix,
        kernel_config["max_perc_undetected"],
    )
    return filtered_chrom_patterns, chrom_pattern_windows


@numba_jit()
def remove_smears(patterns, win_size=8):
    """
    Identify patterns that are too close from each other to exclude them.
    This can happen when patterns smear over several pixels and are detected twice.
    When that happens, the pattern with the highest score in the smear will be kept.

    Parameters
    ----------
    patterns : numpy.array of float
        2D Array of patterns, with 3 columns: row, column and score.
    win_size : int
        The maximum number of pixels at which patterns are considered overlapping.
    
    Returns
    -------
    numpy.array of bool :
        Boolean array indicating which patterns are valid (True values) and
        which are smears (False values)
    """
    p = patterns.copy()
    # Divide each row / col by the window size to serve as a "hash"
    p["round_row"] = p.bin1 // win_size
    p["round_col"] = p.bin2 // win_size
    # Group patterns by row-col combination and retrieve the index of the
    # pattern with the best score in each group
    best_idx = (
        p.groupby(["round_row", "round_col"], sort=False)["score"]
        .idxmax()
        .values
    )
    good_patterns_mask = np.zeros(patterns.shape[0], dtype=bool)
    try:
        good_patterns_mask[best_idx] = True
    except IndexError:
        # no input pattern
        pass
    return good_patterns_mask


@numba_jit()
def picker(mat_conv, precision=None):
    """Pick pixels out of a convolution map
    Given a correlation heat map, pick (i, j) of local maxima
    Parameters
    ----------
    mat_conv : scipy.sparse.coo_matrix
        A float array assigning a correlation to each pixel (i,j)
        with the input kernel (e.g. loops).
    precision : float, optional
        Increasing this value reduces the amount of false positive patterns.
        This is the minimum number of standard deviations above the median of
        correlation coefficients required to consider a pixel as candidate.
    Returns
    -------
    foci_coords : numpy.array of ints
        2D array of coordinates for identified patterns. None is no pattern was
        detected.
    """

    candidate_mat = mat_conv.copy()
    candidate_mat = candidate_mat.tocoo()
    # Compute a threshold from precision arg and set all pixels below to 0
    if precision is None:
        thres = 0
    else:
        thres = np.median(
            mat_conv.data
        ) + precision * ss.median_absolute_deviation(
            mat_conv.data, nan_policy="omit"
        )
    candidate_mat.data[candidate_mat.data < thres] = 0
    candidate_mat.data[candidate_mat.data != 0] = 1
    candidate_mat.eliminate_zeros()

    # Check if at least one candidate pixel was found
    if len(candidate_mat.data) > 0:
        num_foci, labelled_mat = label_connected_pixels_sparse(candidate_mat)
        if num_foci == 0:
            return None
        mat_conv = mat_conv.tocsr()
        # Will hold the coordinates of the best pixel for each focus
        foci_coords = np.zeros([num_foci, 2], int)
        # Iterate over candidate foci
        # NOTE: There can be jumps between foci id due to post labelling
        # removal of single pixel foci in label_connnected_pixels_sparse.
        # This is why we use focus_rank
        for focus_rank, focus_id in enumerate(np.unique(labelled_mat.data)):
            # Remember 1D indices of datapoint in focus
            focus_idx = np.where(labelled_mat.data == focus_id)[0]
            focus_rows, focus_cols = (
                labelled_mat.row[focus_idx],
                labelled_mat.col[focus_idx],
            )
            # Find index of max value within those indices
            focus_pixel_idx = np.argmax(mat_conv[focus_rows, focus_cols])
            # Retrieve row of original index
            original_pixel_idx = focus_idx[focus_pixel_idx]
            focus_pixel_row = labelled_mat.row[original_pixel_idx]
            focus_pixel_col = labelled_mat.col[original_pixel_idx]
            # Save coords of best pixels for this focus in a table
            foci_coords[focus_rank, 0] = focus_pixel_row
            foci_coords[focus_rank, 1] = focus_pixel_col
    else:
        return None
    return foci_coords


@numba_jit()
def label_connected_pixels_sparse(matrix, min_focus_size=2):
    """
    Given a sparse matrix of 1 and 0 values, find
    all foci of continuously neighbouring positive pixels
    and assign them a label according to their focus. Diagonal,
    horizontal and vertical (8-way) adjacency is considered.

    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix
        The input matrix where to label foci. Should be filled with 1
        and 0s.
    min_focus_size: int
        Minimum number of members required to keep a focus.
    
    Returns
    -------
    num_foci : int
        Number of individual foci identified.
    foci_mat : scipy.sparse.coo_matrix:
        The matrix with values replaced by their respective foci
        labels.
    
    Example
    -------
    >>> M.todense()
    array([[1 0 0 0]
           [1 0 1 0]
           [1 0 1 1]
           [0 0 0 0]])
    >>> num_foci, foci_mat = label_foci(M)
    >>> num_foci
    2
    >>>foci_mat.todense()
    array([[1 0 0 0]
           [1 0 2 0]
           [1 0 2 2]
           [0 0 0 0]])
    """

    candidates = matrix.copy()
    n_candidates = len(candidates.data)
    candidates.data = candidates.data.astype(bool)

    def get_1d_foci_transition(m):
        """
        Get a boolean array indicating if the next neighbour of
        each nonzero pixel will be in the same focus.
        """
        # Compute row and col index shifts between candidate pixels.
        # absolute value is used; left/right or up/down does not matter
        row_shift = np.abs(np.diff(m.row))
        col_shift = np.abs(np.diff(m.col))
        # Transform shifts to binary data: shifted by more than 1
        # from previous nonzero pixel ? Invert so that True means we
        # stay in the same focus.
        row_shift[row_shift < 2] = 0
        stay_foci_row = np.invert(row_shift.astype(bool))
        col_shift[col_shift < 2] = 0
        stay_foci_col = np.invert(col_shift.astype(bool))
        # Bitwise AND between row and col "stay arrays" to get
        # indices where neither the rows nor cols shift by more than 1
        # True at index i means: pixel i+1 in same focus as pixel i.
        stay_foci = np.bitwise_and(stay_foci_row, stay_foci_col)
        # Append False since there is no pixel after the last
        stay_foci = np.append(stay_foci, False)
        return stay_foci

    # Since we are using neighborhood with next nonzero pixel,
    # and nonzero pixels are sorted by rows, we need to do
    # the whole operation on the matrix and the transpose
    # (nonzero pixels sorted by columns). This will give both
    # the horizontal and vertical neighbourhood informations.
    stay_foci_hori = get_1d_foci_transition(candidates)
    stay_foci_verti = get_1d_foci_transition(candidates.T.tocsr().tocoo())
    # Candidates are sorted by rows in stay_foci_hori, but by col in
    # stay_foci_verti. We need to reorder stay_foci_verti to have them
    # in same order.
    ori_ids = candidates.copy()
    # store candidate ids in data
    ori_ids.data = np.array(range(len(ori_ids.row)), dtype=np.int)
    trans_ids = ori_ids.T  # Transposed matrix: candidates now sorted by column
    # Order is only updated if we convert types
    trans_ids = trans_ids.tocsr().tocoo()
    # Data can now be used as a transposed to original candidate id converter
    # e.g. if trans_idx.data[2] = 1, that  means the third (2) nonzero
    # value in transpose was the second (1) in original matrix
    # stay_foci_verti = stay_foci_verti[trans_ids.data]
    # Initialize adjacency matrix between candidate pixels
    adj_mat = sp.lil_matrix((n_candidates, n_candidates))
    # Fill adjacency matrix using a stay_foci array

    def fill_adjacency_1d(adj, stay_foci, verti=False):
        """Fills adjacency matrix for all neighborhoods on 1 dimension"""
        for candidate_id, next_candidate_in_focus in enumerate(stay_foci):
            # If the next candidate will also be in focus, add fill connection
            # between current and next candidate in adjacency matrix
            if next_candidate_in_focus:
                if verti:
                    adj_from = trans_ids.data[candidate_id]
                    adj_to = trans_ids.data[candidate_id + 1]
                else:
                    adj_from = candidate_id
                    adj_to = candidate_id + 1
                adj[adj_from, adj_to] = 1
        return adj

    # Add horizontal-adjacency info
    adj_mat = fill_adjacency_1d(adj_mat, stay_foci_hori)
    # Add vertical-adjacency info.
    adj_mat = fill_adjacency_1d(adj_mat, stay_foci_verti, verti=True)
    # Now label foci by finding connected components
    num_foci, foci = sp.csgraph.connected_components(adj_mat)
    foci += 1  # We add 1 so that first spot is not 0
    foci = foci.astype(np.int64)
    # Remove foci with a single pixel
    for focus_num in range(1, num_foci + 1):
        if len(foci[foci == focus_num]) < min_focus_size:
            foci[foci == focus_num] = 0
    # mask small foci for removal
    good_foci = foci > 0
    # generate a new matrix, similar to the input, but where pixel values
    # are the foci ID of the pixel.
    foci_mat = sp.coo_matrix(
        (
            foci[good_foci],
            (candidates.row[good_foci], candidates.col[good_foci]),
        ),
        shape=candidates.shape,
        dtype=np.int64,
    )
    # Update number after removal of small foci
    num_foci = len(np.unique(foci_mat.data))
    return num_foci, foci_mat


@numba_jit()
def xcorr2(signal, kernel, threshold=1e-4):
    """Signal-kernel 2D convolution

    Convolution of a 2-dimensional signal (the contact map) with a kernel
    (the pattern template).

    Parameters
    ----------
    signal: scipy.sparse.csr_matrix
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: array_like
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    threshold : float
        Convolution score below which pixels will be set back to zero to save
        on time and memory.
    Returns
    -------
    out: scipy.sparse.coo_matrix
        Convolution product of signal by kernel.
    """

    sm, sn = signal.shape
    km, kn = kernel.shape

    # Kernel (half) height and width
    kh = (km - 1) // 2
    kw = (kn - 1) // 2

    # Sanity checks
    if sp.issparse(kernel):
        raise ValueError("cannot handle kernel in sparse format")
    if (km > sm) or (sn > sn):
        raise ValueError("cannot have kernel bigger than signal")

    if sp.issparse(signal):
        signal = signal.tocsr()
    # Check of kernel is constant (uniform)
    constant_kernel = np.nan
    if np.allclose(kernel, np.tile(kernel[0, 0], kernel.shape), rtol=1e-08):
        constant_kernel = kernel[0, 0]

    out = sp.csc_matrix((sm - km + 1, sn - kn + 1), dtype=np.float64)

    # Simplified convolution for the special case where kernel is constant:
    if np.isfinite(constant_kernel):
        l_subkernel_sp = sp.diags(
            np.ones(km), np.arange(km), shape=(sn - km + 1, sm), format="csr"
        )
        r_subkernel_sp = sp.diags(
            np.ones(kn), -np.arange(kn), shape=(sn, sm - kn + 1), format="csr"
        )
        out = (l_subkernel_sp @ signal) @ r_subkernel_sp
        out *= constant_kernel
    # Convolution code for general case
    else:
        for kj in range(kn):
            subkernel_sp = sp.diags(
                kernel[:, kj],
                np.arange(km),
                shape=(sn - km + 1, sm),
                format="csr",
            )
            out += subkernel_sp.dot(signal[:, kj : sn - kn + 1 + kj])

    # Set very low pixels to 0
    out.data[out.data < threshold] = 0
    out.eliminate_zeros()

    # Resize matrix: increment rows and cols by half kernel and set shape to input
    # matrix, effectively adding margins.
    out = out.tocoo()
    rows, cols = out.row + kh, out.col + kw
    out = sp.coo_matrix(
        (out.data, (rows, cols)), shape=(sm, sn), dtype=np.float64
    )

    return out


@numba_jit()
def corrcoef2d(
    signal, kernel, max_dist=None, sym_upper=False, scaling="pearson"
):
    """Signal-kernel 2D correlation
    Pearson correlation coefficient between signal and sliding kernel. Convolutes
    the input signal and kernel computes a cross correlation coefficient.

    Parameters
    ----------
    signal : scipy.sparse.csr_matrix
        The input processed Hi-C matrix.
    kernel : numpy.array
        The pattern kernel to use for convolution.
    max_dist : int
        Maximum scan distance, in number of bins from the diagonal. If None, the whole
        matrix is convoluted. Otherwise, pixels further than this distance from the
        diagonal are set to 0 and ignored for performance. Only useful for 
        intrachromosomal matrices.
    sym_upper : False
        Whether the matrix is symmetric and upper triangle. True for intrachromosomal
        matrices.
    scaling : str
        Which metric to use when computing correlation coefficients. Either 'pearson'
        for Pearson correlation, or 'cross' for cross correlation.

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse matrix of correlation coefficients
    """
    # If using only the upper triangle matrix, set diagonals that will
    # overlap the kernel in the lower triangle to their opposite diagonal
    # in the upper triangle
    if sym_upper:
        if sp.issparse(signal):
            signal = signal.tolil()
            for i in range(1, kernel.shape[0]):
                signal.setdiag(signal.diagonal(i), -i)
        else:
            # Full matrix is stored for dense arrays anyway
            # -> make symmetric
            sys.stderr.write("Making dense matrix symmetric.\n")
            signal = signal + np.transpose(signal) - np.diag(np.diag(signal))

    kernel_size = kernel.shape[0] * kernel.shape[1]

    if scaling == "cross":
        # Compute convolution product
        conv = xcorr2(signal, kernel)
        # Generate constant kernel
        kernel1 = np.ones(kernel.shape)
        # Convolute squared signal with constant kernel
        if sp.issparse(signal):
            signal2 = xcorr2(signal.power(2), kernel1)
        else:
            signal2 = xcorr2(signal ** 2, kernel1)
        kernel2 = float(np.sum(kernel ** 2))
        denom = signal2 * kernel2
        if sp.issparse(signal):
            denom = denom.sqrt()
        else:
            denom = np.sqrt(denom)
    elif scaling == "pearson":
        mean_kernel = float(kernel.mean())
        std_kernel = float(kernel.std())
        if not (std_kernel > 0):
            raise ValueError(
                "Cannot have scaling=pearson when kernel"
                "is flat. Use scaling=cross."
            )

        kernel1 = np.ones(kernel.shape)
        mean_signal = xcorr2(signal, kernel1 / kernel_size)
        if sp.issparse(signal):
            std_signal = xcorr2(
                signal.power(2), kernel1 / kernel_size
            ) - mean_signal.power(2)
            std_signal = std_signal.sqrt()
        else:
            std_signal = np.sqrt(
                xcorr2(signal ** 2, kernel1 / kernel_size) - mean_signal ** 2
            )

        conv = xcorr2(signal, kernel / kernel_size) - mean_signal * mean_kernel
        denom = std_signal * std_kernel

    # Since elementwise sparse matrices division is not implemented, compute
    # numerator and denominator and perform division on the 1D array of nonzero
    # values.
    # Get coords of non-zero (nz) values in the numerator
    nz_vals = conv.nonzero()
    # Divide them by corresponding entries in the numerator
    denom = denom.tocsr()
    try:
        conv.data /= denom[nz_vals].A1
    # Case there are no nonzero corrcoef
    except AttributeError:
        pass
    conv.data[conv.data < 0] = 0
    if max_dist is not None:
        # Trim diagonals further than max_scan_distance
        conv = preproc.diag_trim(conv.todia(), max_dist)

    # Only keep the upper triangle
    conv = sp.triu(conv)

    return conv
