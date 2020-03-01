from __future__ import absolute_import
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss
import pathlib
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


def validate_patterns(
    coords,
    matrix,
    conv_mat,
    detectable_bins,
    kernel_matrix,
    max_undetected_perc,
):
    """
    Given a list of pattern coordinates and a contact map, remove patterns in low
    detectability regions or too close to matrix boundaries. Also returns the
    surrounding window of Hi-C contacts around each detected pattern.

    Parameters
    ----------
    coords : numpy.array of ints
        Coordinates of all detected patterns in the sub matrix. One pattern per
        row, the first column is the matrix row, second column is the matrix col.
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
        high, low = p1 - half_h + 1, p1 + half_h
        left, right = p2 - half_w + 1, p2 + half_w
        # Check for out of bounds errors
        if (
            high >= 0
            and low < matrix.shape[0]
            and left >= 0
            and right < matrix.shape[1]
        ):
            # Get bin ids to use in window
            win_rows, win_cols = (range(high, low), range(left, right))
            # Subset window from chrom matrix
            pattern_window = matrix[high:low, left:right].toarray()

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
            tot_zero_pixels = len(pattern_window[pattern_window == 0])
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


def pileup_patterns(pattern_windows):
    """
    Generate a pileup from an stack of pattern windows.

    Parameters
    ----------
    pattern_windows : numpy.array of floats
        3D numpy array of detected windows. Shape is (N, H, W)
        where N is the number of windows, H the height, and W
        the width of each window.
    Returns
    -------
    numpy.array of floats :
        2D numpy array containing the pileup (arithmetic mean) of
        input windows.
    """
    return np.apply_along_axis(np.nanmean, 0, pattern_windows)


def pattern_detector(
    contact_map, kernel_config, kernel_matrix, dump=None, full=False
):
    """
    Detect patterns in a contact map by kernel matching, and extract windows
    around the detected patterns.

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
    dump : str or None
        Folder in which dumps should be generated after each step of the detection
        process. If None, no dump is generated

    Returns
    -------
    filtered_chrom_patterns : numpy.array
        A 2D array of detected patterns with 3 columns: bin1, bin2, score.
    chrom_pattern_windows : numpy array
        A 3D array containing the pile of windows around detected patterns.
    """
    km, kn = kernel_matrix.shape
    kh, kw = (km - 1) // 2, (kn - 1) // 2
    # Define where to save the dump
    save_dump = lambda base, mat: sp.save_npz(
        pathlib.Path(dump) / f"{contact_map.name}_{base}", mat
    )

    # Do not attempt pattern detection unless matrix is larger than the kernel
    if min(contact_map.matrix.shape) <= max(kernel_matrix.shape):
        return None, None

    if full:
        missing_mask = preproc.missing_bins_mask(
            contact_map.matrix.shape,
            valid_rows=contact_map.detectable_bins[0],
            valid_cols=contact_map.detectable_bins[1],
        )
    else:
        missing_mask = None

    # Pattern matching operate here
    mat_conv = corrcoef2d(
        contact_map.matrix.tocsr(),
        kernel_matrix,
        max_dist=contact_map.max_dist,
        sym_upper=not contact_map.inter,
        full=full,
        missing_mask=missing_mask,
    )
    if dump:
        save_dump("03_corrcoef2d", mat_conv)
    # Clean potential missing values
    mat_conv.data[np.isnan(mat_conv.data)] = 0
    # Only keep corrcoefs in scannable range
    if not contact_map.inter:
        mat_conv = preproc.diag_trim(mat_conv.todia(), contact_map.max_dist)
        if dump:
            save_dump("04_diag_trim", mat_conv)
    mat_conv = mat_conv.tocoo()
    mat_conv.eliminate_zeros()

    # Find foci of highly correlated pixels
    chrom_pattern_coords, foci_mat = picker(mat_conv, kernel_config["pearson"])

    if chrom_pattern_coords is None:
        return None, None
    if dump:
        save_dump("05_foci", foci_mat)
    mat = contact_map.matrix.copy()
    det = [d.copy() for d in contact_map.detectable_bins]
    # Zero pad contact and convolution maps and shift missing bins and detected
    # pattern coords before validation if in full mode
    if full:
        mat = mat.tocoo()
        mat = preproc.zero_pad_sparse(mat, kh, kw, fmt="csr")
        mat_conv = preproc.zero_pad_sparse(mat_conv, kh, kw, fmt="csr")
        det[0] += kh
        det[1] += kw
        chrom_pattern_coords[:, 0] += kh
        chrom_pattern_coords[:, 1] += kw

    if not contact_map.inter:
        # Symmetrize first kh / 2 diagonals in the lower triangle to have nicer
        # pileups and do not count them as missing (otherwise all patterns on
        # diagonal would have 50% missing
        big_k = max(kh, kw)
        mat = mat.tolil()
        for i in range(1, big_k):
            mat.setdiag(mat.diagonal(i), -i)

    filtered_chrom_patterns, chrom_pattern_windows = validate_patterns(
        chrom_pattern_coords,
        mat,
        mat_conv.tocsr(),
        contact_map.detectable_bins,
        kernel_matrix,
        kernel_config["max_perc_undetected"],
    )

    # Shift coordinates of detected patterns back if padding was added
    if full:
        filtered_chrom_patterns.bin1 -= kh
        filtered_chrom_patterns.bin2 -= kw
    return filtered_chrom_patterns, chrom_pattern_windows


def remove_neighbours(patterns, win_size=8):
    """
    Identify patterns that are too close from each other to exclude them.
    The pattern with the highest score in the group will be kept.

    Parameters
    ----------
    patterns : numpy.array of float
        2D Array of patterns, with 3 columns: bin1, bin2 and score.
    win_size : int
        The maximum number of pixels at which patterns are considered overlapping.
    
    Returns
    -------
    numpy.array of bool :
        Boolean array indicating which patterns are valid (True values) and
        which are overlapping neighbours (False values)
    """
    # Sort patterns by scores
    sorted_patterns = patterns.copy().sort_values("score", ascending=False)
    # Keep track of patterns to drop
    blacklist = set()
    for i, p in sorted_patterns.iterrows():
        if i not in blacklist:
            close_patterns = np.where(
                (np.abs(sorted_patterns.bin1 - p.bin1) < win_size)
                & (np.abs(sorted_patterns.bin2 - p.bin2) < win_size)
            )[0]
            close_patterns_idx = sorted_patterns.index.values[close_patterns]
            close_patterns_idx = close_patterns_idx[close_patterns_idx != i]
            for idx in close_patterns_idx:
                blacklist.add(idx)
    # Get indices of valid patterns as a boolean mask
    whitelist_mask = np.ones(sorted_patterns.shape[0], dtype=bool)
    whitelist_mask[list(blacklist)] = False
    return whitelist_mask


def picker(mat_conv, pearson):
    """
    Pick coordinates of local maxima in a sparse 2D convolution heatmap. A threshold
    computed based on the pearson argument is applied to the heatmap. All values below
    that threshold are set to 0. The coordinate of the maximum value in each focus is
    returned.

    Parameters
    ----------
    mat_conv : scipy.sparse.coo_matrix of floats
        A 2D sparse convolution heatmap.
    pearson : float
        Minimum correlation coefficient required to consider a pixel as candidate.
        Increasing this value reduces the amount of false positive patterns.
    Returns
    -------
    foci_coords : numpy.array of ints
        2D array of coordinates for identified patterns corresponding to foci maxima.
        None is no pattern was detected.
    labelled_mat : scipy.sparse.coo_matrix
        The map of detected foci. Pixels which were assigned to a focus are given
        an integer as their focus ID. Pixels not assigned to a focus are set to 0.
    """

    candidate_mat = mat_conv.copy()
    candidate_mat = candidate_mat.tocoo()
    candidate_mat.data[candidate_mat.data < pearson] = 0
    candidate_mat.data[candidate_mat.data != 0] = 1
    candidate_mat.eliminate_zeros()

    # Check if at least one candidate pixel was found
    if len(candidate_mat.data) > 0:
        num_foci, labelled_mat = label_foci(candidate_mat)
        num_foci, labelled_mat = filter_foci(labelled_mat)
        if num_foci == 0:
            return None, None
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
        return None, None
    return foci_coords, labelled_mat


def label_foci(matrix):
    """
    Given a sparse matrix of 1 and 0 values, find
    all foci of continuously neighbouring positive pixels
    and assign them a label according to their focus. Horizontal 
    and vertical (4-way) adjacency is considered.

    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix of ints
        The input matrix where to label foci. Should be filled with 1
        and 0s.
    
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
    # step 1 : ensure that coo format is double ranked (row, col).
    # TODO: we could find a trick to avoid this conversion step
    matrix_sp = sp.coo_matrix(sp.csr_matrix(matrix))

    nb_row, nb_col = matrix_sp.shape
    nb_nodes = matrix_sp.nnz

    # step 2: prepare structured array 'coo' for later double sorting
    dtype = [("row", int), ("col", int)]
    coo = np.zeros(matrix_sp.nnz, dtype=dtype)
    coo["row"] = matrix_sp.row
    coo["col"] = matrix_sp.col

    # step 3: find neigbors to the right
    coo_rc = np.argsort(coo, order=["row", "col"])
    row_rc = coo["row"][coo_rc]
    col_rc = coo["col"][coo_rc]

    # sanity check
    if not (np.all(coo_rc == np.arange(len(coo_rc)))):
        raise ValueError("matrix_sp is not properly double sorted")

    # Compute row and col indices shift between right-neighbors
    diff_row_rc = row_rc[1:] - row_rc[:-1]
    diff_col_rc = col_rc[1:] - col_rc[:-1]
    # Right connected pixels are on the same row and neighboring cols
    right_connected_p = (diff_row_rc == 0) & (diff_col_rc == 1)
    right_connected_k = np.where(right_connected_p)[0]

    right_node1 = right_connected_k
    right_node2 = right_connected_k + 1

    # step 4: find neigbors beneath
    coo_cr = np.argsort(coo, order=["col", "row"])
    row_cr = coo["row"][coo_cr]
    col_cr = coo["col"][coo_cr]

    diff_row_cr = row_cr[1:] - row_cr[:-1]
    diff_col_cr = col_cr[1:] - col_cr[:-1]

    lower_connected_p = (diff_row_cr == 1) & (diff_col_cr == 0)
    lower_connected_k = np.where(lower_connected_p)[0]

    cr2rc = np.arange(len(coo_cr))[coo_cr]
    lower_node1 = cr2rc[lower_connected_k]
    lower_node2 = cr2rc[lower_connected_k + 1]

    # step 5: build adjacency matrix
    node1 = np.concatenate([right_node1, lower_node1])
    node2 = np.concatenate([right_node2, lower_node2])
    data = np.ones(len(node1), int)
    adj_mat = sp.coo_matrix((data, (node1, node2)), shape=(nb_nodes, nb_nodes))

    # step 6: find connected components
    num_foci, foci = sp.csgraph.connected_components(adj_mat, directed=False)
    foci_mat = sp.coo_matrix(
        (foci + 1, (matrix_sp.row, matrix_sp.col)), shape=(nb_row, nb_col)
    )

    return num_foci, foci_mat


def filter_foci(foci_mat, min_size=2):
    """
    Given an input matrix of labelled foci (continuous islands of equal nonzero
    values), filters out foci consisting of fewer pixels than min_size.

    Parameters
    ----------
    foci_mat : scipy.sparse.coo_matrix
        Input matrix of labelled foci. Pixels are numbered according to their
        respective foci. Pixels that are not assigned to a focus are 0.
    min_size : int
        Minimum number of pixels required to keep a focus. Pixels belonging to
        smaller foci will be set to 0.
    
    Returns
    -------
    num_filtered : int
        Number of foci remaining after filtering.
    filtered_mat : scipy.sparse.coo_matrix
        Matrix of filtered foci.
    """
    foci_data = foci_mat.data
    focus_id, focus_size = np.unique(foci_data, return_counts=True)
    # Remove foci smaller than min_size
    small_foci = focus_id[focus_size < min_size]
    # mask small foci for removal
    for focus_id in small_foci:
        foci_data[foci_data == focus_id] = 0
    # Copy input matrix and replace data with filtered foci
    filtered_mat = foci_mat.copy()
    filtered_mat.data = foci_data
    # Force updating to make explicit zeros implicit
    filtered_mat.eliminate_zeros()
    num_filtered = len(focus_size[focus_size >= min_size])

    return num_filtered, filtered_mat


def xcorr2(signal, kernel, threshold=1e-4):
    """
    Signal-kernel 2D convolution

    Convolution of a 2-dimensional signal (the contact map) with a kernel
    (the pattern template).

    Parameters
    ----------
    signal: scipy.sparse.csr_matrix or numpy.array of floats
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: numpy.array of floats
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    threshold : float
        Convolution score below which pixels will be set back to zero to save
        on time and memory.
    Returns
    -------
    out: scipy.sparse.csr_matrix or numpy.array
        Convolution product of signal by kernel. Same type as the input signal.
    """

    if sp.issparse(signal):
        conv = _xcorr2_sparse(signal.tocsr(), kernel, threshold=threshold)
    else:
        conv = _xcorr2_dense(signal, kernel, threshold=threshold)
    return conv


def corrcoef2d(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    scaling="pearson",
    full=False,
    missing_mask=None,
):
    """
    Signal-kernel 2D correlation
    Pearson correlation coefficient between signal and sliding kernel. Convolutes
    the input signal and kernel computes a cross correlation coefficient.

    Parameters
    ----------
    signal : scipy.sparse.csr_matrix or numpy.array
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
    missing_mask : scipy.sparse.coo_matrix of ints
        Matrix defining which pixels are missing (1) or not (0).
    full : bool
        If True, convolutions will be made in 'full' mode; the matrix is first
        padded with margins to allow scanning to the edges, and missing bins are
        also masked to exclude them when computing correlation scores. Computationally
        intensive
    missing_mask : scipy.sparse.csr_matrix of bool or None
        Mask matrix denoting missing bins, where missing is denoted as True and
        valid as False. Can be None to ignore missing bin information. Only taken
        into account when full=True.

    Returns
    -------
    scipy.sparse.csr_matrix or numpy.array
        The sparse matrix of correlation coefficients. Same type as the input signal.
    """

    if min(kernel.shape) >= max(signal.shape):
        raise ValueError("cannot have kernel bigger than signal")

    if sp.issparse(signal):
        corr = _corrcoef2d_sparse(
            signal,
            kernel,
            max_dist=max_dist,
            sym_upper=sym_upper,
            scaling=scaling,
            full=full,
            missing_mask=missing_mask,
        )
    else:
        if full:
            raise NotImplementedError(
                "Full convolution is not available available for dense matrices yet."
            )
        corr = _corrcoef2d_dense(
            signal,
            kernel,
            max_dist=max_dist,
            sym_upper=sym_upper,
            scaling=scaling,
        )
    return corr


def _xcorr2_sparse(signal, kernel, threshold=1e-4):
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
    out: scipy.sparse.csr_matrix
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
    if not sp.issparse(signal):
        raise ValueError("cannot handle signal in dense format")
    # Check of kernel is constant (uniform)
    constant_kernel = np.nan
    if np.allclose(kernel, np.tile(kernel[0, 0], kernel.shape), rtol=1e-08):
        constant_kernel = kernel[0, 0]

    out = sp.csc_matrix((sm - km + 1, sn - kn + 1), dtype=np.float64)

    # Simplified convolution for the special case where kernel is constant:
    if np.isfinite(constant_kernel):
        l_subkernel_sp = sp.diags(
            constant_kernel * np.ones(km),
            np.arange(km),
            shape=(sm - km + 1, sm),
            format="dia",
        )
        r_subkernel_sp = sp.diags(
            np.ones(kn), -np.arange(kn), shape=(sn, sn - kn + 1), format="dia"
        )
        out = (l_subkernel_sp @ signal) @ r_subkernel_sp
    # Convolution code for general case
    else:
        for kj in range(kn):
            subkernel_sp = sp.diags(
                kernel[:, kj],
                np.arange(km),
                shape=(sm - km + 1, sm),
                format="csr",
            )
            out += subkernel_sp.dot(signal[:, kj : sn - kn + 1 + kj])

    # Set very low pixels to 0
    out.data[out.data < threshold] = 0
    out.eliminate_zeros()

    # Resize matrix: increment rows and cols by half kernel and set shape to input
    # matrix, effectively adding margins.
    out = out.tocoo()
    out = preproc.zero_pad_sparse(out, margin_h=kw, margin_v=kh, fmt="csr")

    return out


def _xcorr2_dense(signal, kernel, threshold=1e-4):
    """Signal-kernel 2D convolution

    Convolution of a 2-dimensional signal (the contact map) with a kernel
    (the pattern template).

    Parameters
    ----------
    signal: numpy.array
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: array_like
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    threshold : float
        Convolution score below which pixels will be set back to zero to save
        on time and memory.
    Returns
    -------
    out: numpy.array
        Convolution product of signal by kernel.
    """
    sm, sn = signal.shape
    km, kn = kernel.shape
    # Kernel (half) height and width
    kh = (km - 1) // 2
    kw = (kn - 1) // 2
    constant_kernel = np.nan

    out_wo_margin = np.zeros([sm - km + 1, sn - kn + 1])
    # Simplified convolution for the special case where kernel is constant:
    if np.isfinite(constant_kernel):
        l_subkernel_sp = sp.diags(
            np.ones(km), np.arange(km), shape=(sm - km + 1, sm), format="csr"
        )
        r_subkernel_sp = sp.diags(
            np.ones(kn), -np.arange(kn), shape=(sn, sn - kn + 1), format="csr"
        )
        out_wo_margin = (l_subkernel_sp @ signal) @ r_subkernel_sp
        out_wo_margin *= constant_kernel
    # Convolution code for general case
    else:
        for kj in range(kn):
            subkernel_sp = sp.diags(
                kernel[:, kj],
                np.arange(km),
                shape=(sm - km + 1, sm),
                format="csr",
            )
            out_wo_margin += subkernel_sp.dot(signal[:, kj : sn - kn + 1 + kj])

    # Add margins of zeros where kernel overlaps edges
    out = np.zeros([sm, sn])
    out[kh:-kh, kw:-kw] = out_wo_margin
    # Set very low pixels to 0
    out[out < threshold] = 0
    return out


def _corrcoef2d_sparse(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    scaling="pearson",
    full=False,
    missing_mask=None,
):
    """Implementation of signal-kernel 2D correlation for sparse matrices
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
        for Pearson correlation or None for basic convolution.
    missing_mask : scipy.sparse.coo_matrix of ints
        Matrix defining which pixels are missing (1) or not (0). Only used with
        mode='full'.
    mode : str
        Convolution mode. Can be either "standard" or "full"

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse matrix of correlation coefficients
    """
    mk, nk = kernel.shape
    ms, ns = signal.shape

    # Generate constant kernel
    kernel1 = np.ones(kernel.shape)
    kernel_size = mk * nk
    # In full mode, we compute the convolution with all pixels in the input
    # signal. We need to add a margin around the input to allow the kernel to
    # be centered on the edges.
    if full:
        # Create a vertical margin and use it to pad the signal
        tmp = sp.csr_matrix((mk - 1, ns))
        signal = sp.vstack([tmp, signal, tmp], format=signal.format)
        # Same for the horizontal margin
        tmp = sp.csr_matrix((ms + 2 * (mk - 1), nk - 1))
        signal = sp.hstack([tmp, signal, tmp], format=signal.format)
        # If a missing mask was specified, use it
        if missing_mask is not None:
            # Add margins around missing mask.
            missing_mask = preproc.make_exterior_frame(
                missing_mask,
                kernel.shape,
                sym_upper=sym_upper,
                max_dist=max_dist,
            )
            # Safety check to make sure mask matches signal
            preproc.check_ismissing(signal, missing_mask)

    # Plain old convolution
    if scaling is None:
        conv = xcorr2(signal, kernel)

    elif scaling == "pearson":
        if missing_mask is None:
            kernel_mean = float(kernel.mean())
            kernel_std = float(kernel.std())
            if not (kernel_std > 0):
                raise ValueError(
                    "Cannot have scaling=pearson when kernel"
                    "is flat. Use scaling=cross."
                )
            signal_mean = xcorr2(signal, kernel1 / kernel_size)
            signal_std = (
                xcorr2(signal.power(2), kernel1 / kernel_size)
                - signal_mean.power(2)
            ).sqrt()
            conv = (
                xcorr2(signal, kernel / kernel_size)
                - signal_mean * kernel_mean
            )
            conv = preproc.sparse_division(conv, signal_std * kernel_std)
        else:
            # bigmat = lambda x: sp.coo_matrix(np.ones(missing_mask.shape) * x)
            kernel_sum = np.sum(kernel)
            kernel_mean = kernel_sum / kernel_size
            kernel_std = float(kernel.std())
            kernel2_sum = np.sum(kernel ** 2)
            kernel2_mean = kernel2_sum / kernel_size
            # Compute convolution of uniform kernel with missing mask to get overlap
            # information into pearson later
            ker1_coo = _xcorr2_sparse(missing_mask, kernel1).tocoo()
            # Make sure there is no signal in masked regions
            missing_mask = missing_mask.tocoo()
            signal[missing_mask.row, missing_mask.col] = 0.0
            missing_mask = missing_mask.tocsr()
            # Basically, true if there is a margin
            if len(ker1_coo.data) > 0:
                # TODO: handle cases where kernel_size_wm = 0 !!!
                # Compute pearson terms with margin taken into account
                kernel_size_wm = kernel_size - ker1_coo.data
                ker1_coo_row, ker1_coo_col = ker1_coo.row, ker1_coo.col
                ker1_coo = None
                del ker1_coo  # Spare memory, we only need coords
                ker_csr = _xcorr2_sparse(missing_mask, kernel)
                # kernel size (w)ith (m)issing data
                kernel_mean_wm = (
                    kernel_sum - np.array(ker_csr[ker1_coo_row, ker1_coo_col])
                ) / kernel_size_wm
                ker_coo = None
                del ker_coo
                ker2_csr = _xcorr2_sparse(missing_mask, kernel ** 2)
                kernel2_mean_wm = (
                    kernel2_sum
                    - np.array(ker2_csr[ker1_coo_row, ker1_coo_col])
                ) / kernel_size_wm
                ker2_csr = None
                del ker2_csr
            else:
                kernel_size_wm = kernel_size
                kernel_mean_wm = kernel_mean
                kernel2_mean_wm = kernel2_mean

            # Compute signal mean by convoluting uniform kernel and dividing by mean
            signal_mean = xcorr2(signal, kernel1 / kernel_size).tocsr()
            # Multiply each pixel in mean signal by the proportion of kernel
            # that's not in a margin
            signal_mean[ker1_coo_row, ker1_coo_col] = (
                np.array(signal_mean[ker1_coo_row, ker1_coo_col])
                * (kernel_size / kernel_size_wm)
            ).flat

            # Same for the denominator
            denom = (xcorr2(signal.power(2), kernel1) / kernel_size).tocsr()
            denom[ker1_coo_row, ker1_coo_col] = (
                np.array(denom[ker1_coo_row, ker1_coo_col])
                * (kernel_size / kernel_size_wm)
            ).flat
            denom = (denom - signal_mean.power(2)) * (
                kernel2_mean - kernel_mean ** 2
            )
            denom[ker1_coo_row, ker1_coo_col] = (
                (
                    np.array(denom[ker1_coo_row, ker1_coo_col])
                    / (kernel2_mean - kernel_mean ** 2)
                )
                * (kernel2_mean_wm - kernel_mean_wm ** 2)
            ).flat
            denom = denom.sqrt()
            # 2 tricks to improve numeric stability:
            # Remove very low correlations in the denominators
            denom[abs(denom) < 1e-10] = 0.0
            # Exclude correlation coefficients computed on too few kernel values
            # we select 25% of kernel, as in 50% width and 50% height
            unstable_coeffs = kernel_size_wm < kernel_size / 4
            unstable_rows = ker1_coo_row[unstable_coeffs]
            unstable_cols = ker1_coo_col[unstable_coeffs]
            denom[unstable_rows, unstable_cols] = 0
            denom.eliminate_zeros()
            conv = signal_mean * kernel_mean
            conv[ker1_coo_row, ker1_coo_col] = (
                np.array(conv[ker1_coo_row, ker1_coo_col])
                * (kernel_mean_wm / kernel_mean)
                * (kernel_size_wm / kernel_size)
            ).flat
            conv = xcorr2(signal, kernel) / kernel_size - conv
            conv[ker1_coo_row, ker1_coo_col] = (
                np.array(conv[ker1_coo_row, ker1_coo_col])
                * (kernel_size / kernel_size_wm)
            ).flat
            conv.eliminate_zeros()
            denom.eliminate_zeros()
            conv = preproc.sparse_division(conv.tocoo(), denom.tocoo())
    else:
        raise ValueError("scaling must be either None or pearson.")
    if (max_dist is not None) and sym_upper:
        # Trim diagonals further than max_scan_distance
        conv = preproc.diag_trim(conv.todia(), max_dist)

    if sym_upper:
        conv = sp.triu(conv)
    conv = conv.tocoo()
    conv.data[~np.isfinite(conv.data)] = 0.0
    conv.data[conv.data < 0] = 0.0
    conv.eliminate_zeros()
    conv = conv.tocsr()
    if full:
        conv = conv.tocsr()[mk - 1 : -mk + 1, nk - 1 : -nk + 1]
    return conv


def _corrcoef2d_dense(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    scaling="pearson",
    mode="valid",
):
    """Implementation of signal-kernel 2D correlation for dense matrices
    Pearson correlation coefficient between signal and sliding kernel. Convolutes
    the input signal and kernel computes a cross correlation coefficient.

    Parameters
    ----------
    signal : numpy.array
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
        for Pearson correlation or None for basic convolution.

    Returns
    -------
    numpy.array
        The sparse matrix of correlation coefficients
    """

    # Convert numpy matrices to array to avoid operator overloading
    if isinstance(signal, np.matrix):
        signal = np.array(signal)
    if isinstance(kernel, np.matrix):
        kernel = np.array(kernel)

    kernel_size = kernel.shape[0] * kernel.shape[1]
    kernel1 = np.ones(kernel.shape)
    # Plain old convolution
    if scaling is None:
        conv = xcorr2(signal, kernel)
    # Pearson correlation
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

        std_signal = (
            xcorr2(signal ** 2, kernel1 / kernel_size) - mean_signal ** 2
        )
        std_signal = np.sqrt(std_signal)
        conv = xcorr2(signal, kernel / kernel_size) - mean_signal * mean_kernel
        denom = std_signal * std_kernel

        conv /= denom
    else:
        raise ValueError("scaling must be either None or pearson.")

    if (max_dist is not None) and sym_upper:
        # Trim diagonals further than max_scan_distance
        conv = preproc.diag_trim(conv, max_dist)

    if sym_upper:
        conv = np.triu(conv)
    conv[~np.isfinite(conv)] = 0.0
    conv[conv < 0] = 0.0
    return conv
