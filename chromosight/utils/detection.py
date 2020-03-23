from __future__ import absolute_import
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pathlib
import warnings
import chromosight.utils.preprocessing as preproc
import chromosight.utils.stats as cus

warnings.filterwarnings("ignore")


def validate_patterns(
    coords,
    matrix,
    conv_mat,
    detectable_bins,
    kernel_matrix,
    max_undetected_perc,
    drop=True,
):
    """
    Given a list of pattern coordinates and a contact map, remove patterns in
    low detectability regions or too close to matrix boundaries. Also returns
    the surrounding window of Hi-C contacts around each detected pattern.

    Parameters
    ----------
    coords : numpy.array of ints
        Coordinates of all detected patterns in the sub matrix. One pattern
        per row, the first column is the matrix row, second column is the
        matrix col.
    matrix : scipy.sparse.csr_matrix
        Hi-C contact map of the sub matrix.
    conv_mat : scipy.sparse.csr_matrix
        Convolution product of the kernel with the Hi-C sub matrix.
    detectable_bins : list of numpy.array
        List of two 1D numpy arrays of ints representing ids of detectable
        rows and columns, respectively.
    kernel_matrix : numpy.array of float
        The kernel that was used for pattern detection on the Hi-C matrix.
    max_undetected_perc : float
        Proportion of undetectable pixels allowed in a pattern window to
        consider it valid.
    drop : bool
        Whether to discard pattern coordinates and windows from patterns which
        fall outside the matrix or do not pass validation. If those are kept,
        they will be given a score of np.nan and their windows will be filled
        with np.nan.

    Returns
    -------
    filtered_coords : pandas.DataFrame
        Table of coordinates that passed the filters. The dataframe has 3:
        columns: bin1 (rows), bin2 (col) and score (the correlation
        coefficient).
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
            "score": np.full(coords.shape[0], np.nan),
        }
    )
    # validated_coords = np.append(coords, np.zeros((coords.shape[0], 1)), 1)
    # Initialize structure to store pattern windows
    pattern_windows = np.full(
        (coords.shape[0], win_h, win_w), np.nan
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

            # Number of undetected pixels is "bad rows area" + "bad cols
            # area" - "bad rows x bad cols intersection"
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
    if drop:
        blacklist = np.array(blacklist)
        blacklist_mask = np.zeros(coords.shape[0], dtype=bool)
        if len(blacklist):
            blacklist_mask[blacklist] = True
        filtered_coords = validated_coords.loc[~blacklist_mask, :]
        filtered_windows = pattern_windows[~blacklist_mask, :, :]
    else:
        filtered_coords = validated_coords
        filtered_windows = pattern_windows

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
    return np.apply_along_axis(np.nanmedian, 0, pattern_windows)


def pattern_detector(
    contact_map,
    kernel_config,
    kernel_matrix,
    coords=None,
    dump=None,
    full=False,
    tsvd=None,
):
    """
    Detect patterns in a contact map by kernel matching, and extract windows
    around the detected patterns. If coordinates are provided, detection is skipped and
    windows are extracted around those coordinates.

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
    coords : numpy.array of ints or None
        A table with coordinates of patterns, with one pattern per row
        and 2 columns being the row and column number of the pattern in
        the input contact map. If this is provided, detection is skipped
        and quantification is performed on those coordinates.
    dump : str or None
        Folder in which dumps should be generated after each step of the
        detection process. If None, no dump is generated
    tsvd : float or None
        If a float between 0 and 1 is given, the input kernel is factorised
        using truncated SVD, keeping enough singular vectors to retain this
        proportion of information. Factorisation speeds up convolution at
        the cost of a loss of information. If the number of singular vectors
        required to retain the desired information isDisabled by default.

    Returns
    -------
    filtered_chrom_patterns : pandas.DataFrame
        A table of detected patterns with 4 columns: bin1, bin2, score, qvalue.
    chrom_pattern_windows : numpy array
        A 3D array containing the pile of windows around detected patterns.
    """
    km, kn = kernel_matrix.shape
    kh, kw = (km - 1) // 2, (kn - 1) // 2
    # Define where to save the dump
    save_dump = lambda base, mat: sp.save_npz(
        pathlib.Path(dump) / f"{contact_map.name}_{base}", mat
    )

    # Define type of analysis.
    run_mode = "detect" if coords is None else "quantify"

    # Do not attempt pattern detection unless matrix is larger than the kernel
    if min(contact_map.matrix.shape) <= max(kernel_matrix.shape):
        return None, None

    if full:
        missing_mask = preproc.make_missing_mask(
            contact_map.matrix.shape,
            valid_rows=contact_map.detectable_bins[0],
            valid_cols=contact_map.detectable_bins[1],
            max_dist=contact_map.max_dist,
            sym_upper=not contact_map.inter,
        )
    else:
        missing_mask = None

    # Pattern matching operate here
    mat_conv, mat_log10_pvals = normxcorr2(
        contact_map.matrix.tocsr(),
        kernel_matrix,
        max_dist=contact_map.max_dist,
        sym_upper=not contact_map.inter,
        full=full,
        missing_mask=missing_mask,
        tsvd=tsvd,
        pval=True,
    )
    if dump:
        save_dump("03_normxcorr2", mat_conv)
    # Clean potential missing values
    mat_conv.data[np.isnan(mat_conv.data)] = 0
    # Only keep corrcoefs in scannable range
    if not contact_map.inter:
        mat_conv = preproc.diag_trim(mat_conv.tocsr(), contact_map.max_dist)
        if dump:
            save_dump("04_diag_trim", mat_conv)
    mat_conv = mat_conv.tocoo()
    mat_conv.eliminate_zeros()

    # Only attempt detection if no input coordinates were given
    if run_mode == "detect":
        # Find foci of highly correlated pixels and pick local maxima
        coords, foci_mat = picker(mat_conv, kernel_config["pearson"],)
        # If nothing was detected, no point in resuming
        if coords is None:
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
        coords[:, 0] += kh
        coords[:, 1] += kw

    if not contact_map.inter:
        # set the first kh / 2 diagonals in the lower triangle to NaN
        # so that pileups do not count them
        big_k = max(km, kn)
        mat = mat.tocsr()
        mat += sp.diags(
            np.full(big_k, np.nan),
            -np.arange(1, big_k + 1),
            shape=mat.shape,
            format="csr",
        )

    # Extract windows around coordinates and assign a correlation
    # to each pattern. In detection mode, we drop invalid patterns
    # in quantification mode, all input patterns are returned.
    filtered_coords, filtered_windows = validate_patterns(
        coords,
        mat,
        mat_conv.tocsr(),
        contact_map.detectable_bins,
        kernel_matrix,
        kernel_config["max_perc_undetected"],
        drop=True if run_mode == "detect" else False,
    )

    # Shift coordinates of detected patterns back if padding was added
    if full:
        filtered_coords.bin1 -= kh
        filtered_coords.bin2 -= kw

    try:
        filtered_coords["pvalue"] = mat_log10_pvals[
            filtered_coords.bin1, filtered_coords.bin2
        ].A1
    # No coordinate passed the validation filters
    except AttributeError:
        filtered_coords["pvalue"] = None
    # Remove log10 transform and correct p-values for multiple testing
    filtered_coords["pvalue"] = 10 ** filtered_coords["pvalue"]
    return filtered_coords, filtered_windows


def remove_neighbours(patterns, win_size=8):
    """
    Identify patterns that are too close from each other to exclude them.
    The pattern with the highest score in the group will be kept.

    Parameters
    ----------
    patterns : numpy.array of float
        2D Array of patterns, with 3 columns: bin1, bin2 and score.
    win_size : int
        The maximum number of pixels at which patterns are considered
        overlapping.

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
    Pick coordinates of local maxima in a sparse 2D convolution heatmap. A
    threshold computed based on the pearson argument is applied to the
    heatmap. All values below that threshold are set to 0. The coordinate
    of the maximum value in each focus is returned.

    Parameters
    ----------
    mat_conv : scipy.sparse.coo_matrix of floats
        A 2D sparse convolution heatmap.
    pearson : float
        Minimum correlation coefficient required to consider a pixel as
        candidate. Increasing this value reduces the amount of false
        positive patterns.
    Returns
    -------
    foci_coords : numpy.array of ints
        2D array of coordinates for identified patterns corresponding to
        foci maxima. None is no pattern was detected.
    labelled_mat : scipy.sparse.coo_matrix
        The map of detected foci. Pixels which were assigned to a focus are
        given an integer as their focus ID. Pixels not assigned to a focus
        are set to 0.
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


def xcorr2(signal, kernel, threshold=1e-4, tsvd=None):
    """
    Cross correlate a dense or sparse 2D signal with a dense 2D kernel.

    Parameters
    ----------
    signal: scipy.sparse.csr_matrix or numpy.array of floats
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: numpy.array of floats
        A 2-dimensional numpy array Mk x Nk acting as the pattern template.
    threshold : float
        Convolution score below which pixels will be set back to zero to save
        on time and memory.
    tsvd : float or None
        If a float between 0 and 1 is given, the input kernel is factorised
        using truncated SVD, keeping enough singular vectors to retain this
        proportion of information. Factorisation speeds up convolution at
        the cost of a loss of information. If the number of singular vectors
        required to retain the desired information isDisabled by default.
    -------
    out: scipy.sparse.csr_matrix or numpy.array
        Convolution product of signal by kernel. Same type as the input signal.
    """
    if tsvd is not None:
        kernel = preproc.factorise_kernel(kernel, prop_info=tsvd)
    if sp.issparse(signal):
        conv = _xcorr2_sparse(signal.tocsr(), kernel, threshold=threshold)
    else:
        conv = _xcorr2_dense(signal, kernel, threshold=threshold)
    return conv


def _xcorr2_sparse(signal, kernel, threshold=1e-6):
    """
    Cross correlate a sparse 2D signal with a dense 2D kernel.

    Parameters
    ----------
    signal: scipy.sparse.csr_matrix
        A 2-dimensional numpy array Ms x Ns acting as the detrended Hi-C map.
    kernel: numpy.array of floats or tuple of numpy.arrays
        A 2-dimensional numpy array Mk x Nk acting as the pattern template. Can
        also be a factorised kernel.
    threshold : float
        Convolution score below which pixels will be set back to zero to save
        on time and memory.
    Returns
    -------
    out: scipy.sparse.csr_matrix
        Convolution product of signal by kernel.
    """
    sm, sn = signal.shape

    if type(kernel) is tuple:
        kernel_l, kernel_r = kernel
        km = kernel_l.shape[0]
        kn = kernel_r.shape[1]
        if kernel_l.shape[1] != kernel_r.shape[0]:
            raise ValueError("Kernel factorisation is invalid")
        n_factors = kernel_l.shape[1]
        for f in range(n_factors):
            subkernel_l = sp.diags(
                kernel_l[:, f],
                np.arange(km),
                shape=(sm - km + 1, sm),
                format="dia",
            )
            subkernel_r = sp.diags(
                kernel_r[f, :],
                -np.arange(kn),
                shape=(sn, sn - kn + 1),
                format="dia",
            )
            if f == 0:
                out = (subkernel_l @ signal) @ subkernel_r
            else:
                out += (subkernel_l @ signal) @ subkernel_r
    else:
        km, kn = kernel.shape

        # Sanity checks
        if sp.issparse(kernel):
            raise ValueError("cannot handle kernel in sparse format")
        if not sp.issparse(signal):
            raise ValueError("cannot handle signal in dense format")
        # Check of kernel is constant (uniform)
        constant_kernel = np.nan
        if np.allclose(
            kernel, np.tile(kernel[0, 0], kernel.shape), rtol=1e-08
        ):
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
                np.ones(kn),
                -np.arange(kn),
                shape=(sn, sn - kn + 1),
                format="dia",
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
    out.data[np.abs(out.data) < threshold] = 0
    out.eliminate_zeros()

    # Resize matrix to original dimensions
    out = preproc.zero_pad_sparse(
        out, margin_h=(kn - 1) // 2, margin_v=(km - 1) // 2, fmt="csr"
    )
    return out


def _xcorr2_dense(signal, kernel, threshold=1e-6):
    """Cross correlate a dense 2D signal with a dense 2D kernel.

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
    if type(kernel) is tuple:
        kernel_l, kernel_r = kernel
        km = kernel_l.shape[0]
        kn = kernel_r.shape[1]
        if kernel_l.shape[1] != kernel_r.shape[0]:
            raise ValueError("Kernel factorisation is invalid")
        n_factors = kernel_l.shape[1]
        for f in range(n_factors):
            subkernel_l = sp.diags(
                kernel_l[:, f],
                np.arange(km),
                shape=(sm - km + 1, sm),
                format="dia",
            )
            subkernel_r = sp.diags(
                kernel_r[f, :],
                -np.arange(kn),
                shape=(sn, sn - kn + 1),
                format="dia",
            )
            if f == 0:
                out = (subkernel_l @ signal) @ subkernel_r
            else:
                out += (subkernel_l @ signal) @ subkernel_r
    else:
        km, kn = kernel.shape
        # Kernel (half) height and width
        constant_kernel = np.nan
        if np.allclose(
            kernel, np.tile(kernel[0, 0], kernel.shape), rtol=1e-08
        ):
            constant_kernel = kernel[0, 0]
        out_wo_margin = np.zeros([sm - km + 1, sn - kn + 1])
        # Simplified convolution for the special case where kernel is constant:
        if np.isfinite(constant_kernel):
            l_subkernel_sp = sp.diags(
                constant_kernel * np.ones(km),
                np.arange(km),
                shape=(sm - km + 1, sm),
                format="dia",
            )
            r_subkernel_sp = sp.diags(
                np.ones(kn),
                -np.arange(kn),
                shape=(sn, sn - kn + 1),
                format="dia",
            )
            out_wo_margin = (l_subkernel_sp @ signal) @ r_subkernel_sp
        # convolution code for general case
        else:
            for kj in range(kn):
                subkernel_sp = sp.diags(
                    kernel[:, kj],
                    np.arange(km),
                    shape=(sm - km + 1, sm),
                    format="csr",
                )
                out_wo_margin += (
                    subkernel_sp @ signal[:, kj : sn - kn + 1 + kj]
                )

    kh = (km - 1) // 2
    kw = (kn - 1) // 2
    # Add margins of zeros where kernel overlaps edges
    out = np.zeros([sm, sn])
    out[kh:-kh, kw:-kw] = out_wo_margin
    # Set very low pixels to 0
    out[np.abs(out) < threshold] = 0
    return out


def normxcorr2(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    full=False,
    missing_mask=None,
    missing_tol=0.75,
    tsvd=None,
    pval=False,
):
    """
    Computes the normalized cross-correlation of a dense or sparse signal and a
    dense kernel. The resulting matrix contains Pearson correlation
    coefficients.

    Parameters
    ----------
    signal : scipy.sparse.csr_matrix or numpy.array
        The input processed Hi-C matrix.
    kernel : numpy.array
        The pattern kernel to use for convolution.
    max_dist : int
        Maximum scan distance, in number of bins from the diagonal. If None,
        the whole matrix is convoluted. Otherwise, pixels further than this
        distance from the diagonal are set to 0 and ignored for performance.
        Only useful for intrachromosomal matrices.
    sym_upper : False
        Whether the matrix is symmetric and upper triangle. True for
        intrachromosomal
        matrices.
    missing_mask : scipy.sparse.coo_matrix of ints
        Matrix defining which pixels are missing (1) or not (0).
    full : bool
        If True, convolutions will be made in 'full' mode; the matrix is first
        padded with margins to allow scanning to the edges, and missing bins
        are also masked to exclude them when computing correlation scores.
        Computationally intensive
    missing_mask : scipy.sparse.csr_matrix of bool or None
        Mask matrix denoting missing bins, where missing is denoted as True and
        valid as False. Can be None to ignore missing bin information. Only
        taken into account when full=True.
    missing_tol : float
        Proportion of missing values allowed in windows to keep the correlation
        coefficients.
    tsvd : float or None
        If a float between 0 and 1 is given, the input kernel is factorised
        using truncated SVD, keeping enough singular vectors to retain this
        proportion of information. Factorisation speeds up convolution at
        the cost of a loss of information. If the number of singular vectors
        required to retain the desired information isDisabled by default.
    pval : bool
        Whether to return a matrix of p-values.

    Returns
    -------
    scipy.sparse.csr_matrix or numpy.array
        The sparse matrix of correlation coefficients. Same type as the input
        signal.
    scipy.sparse.csr_matrix or numpy.array or None
        A map of Benjamini-Hochberg corrected p-values (q-values). Same type as
        the input signal. If pval=False, this will be None.
    """

    if missing_mask is not None:
        if not sp.issparse(missing_mask):
            raise ValueError("Missing mask must be a sparse matrix.")
        if not signal.shape == missing_mask.shape:
            raise ValueError(
                "Signal and missing mask do not have the same shape"
            )
        if missing_mask.dtype != bool:
            raise ValueError(
                f"Missing mask dtype is {missing_mask.dtype}. Should be bool."
            )

        if min(kernel.shape) >= max(signal.shape):
            raise ValueError("cannot have kernel bigger than signal")
        preproc.check_missing_mask(signal, missing_mask)
        if missing_mask.nnz == 0:
            missing_mask = None

    if sp.issparse(kernel):
        raise ValueError("cannot handle kernel in sparse format")

    if not (kernel.std() > 0):
        raise ValueError("Cannot have flat kernel.")
    if sp.issparse(signal):
        corr, pvals = _normxcorr2_sparse(
            signal,
            kernel,
            max_dist=max_dist,
            sym_upper=sym_upper,
            full=full,
            missing_mask=missing_mask,
            missing_tol=missing_tol,
            tsvd=tsvd,
            pval=pval,
        )
    else:
        corr, pvals = _normxcorr2_dense(
            signal,
            kernel,
            max_dist=max_dist,
            sym_upper=sym_upper,
            full=full,
            missing_tol=missing_tol,
            missing_mask=missing_mask,
            tsvd=tsvd,
            pval=pval,
        )
    return corr, pvals


def _normxcorr2_sparse(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    full=False,
    missing_mask=None,
    missing_tol=0.75,
    tsvd=None,
    pval=False,
):
    """Computes the normalized cross-correlation of a sparse signal and a
    dense kernel. The resulting sparse matrix contains Pearson correlation
    coefficients.

    Parameters
    ----------
    signal : scipy.sparse.csr_matrix
        The input processed Hi-C matrix.
    kernel : numpy.array
        The pattern kernel to use for convolution. Can be a factorised kernel
        stored in a tuple.
    max_dist : int
        Maximum scan distance, in number of bins from the diagonal. If None,
        the whole matrix is convoluted. Otherwise, pixels further than this
        distance from the diagonal are set to 0 and ignored for performance.
        Only useful for intrachromosomal matrices.
    sym_upper : False
        Whether the matrix is symmetric and upper triangle. True for
        intrachromosomal matrices.
    missing_mask : scipy.sparse.csr_matrix of bools
        Matrix defining which pixels are missing (True) or not (False).
    missing_tol : float
        Proportion of missing values allowed in windows to keep the correlation
        coefficients.
    full : bool
        Whether to run in 'full' mode, which means enclosing the signal in an
        exterior frame and computing the correlation up to the edges.
    tsvd : float or None
        If a float between 0 and 1 is given, the input kernel is factorised
        using truncated SVD, keeping enough singular vectors to retain this
        proportion of information. Factorisation speeds up convolution at
        the cost of a loss of information. If the number of singular vectors
        required to retain the desired information isDisabled by default.
    pval : bool
        Whether to return a matrix of p-values.

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
        # TODO: delegate signal framing to frame
        tmp = sp.csr_matrix((mk - 1, ns))
        framed_sig = sp.vstack([tmp, signal, tmp], format=signal.format)
        # Same for the horizontal margin
        tmp = sp.csr_matrix((ms + 2 * (mk - 1), nk - 1))
        framed_sig = sp.hstack([tmp, framed_sig, tmp], format=signal.format)
        # If a missing mask was specified, use it
        if missing_mask is not None:
            # Add margins around missing mask.
            framed_missing_mask = preproc.frame_missing_mask(
                missing_mask,
                kernel.shape,
                sym_upper=sym_upper,
                max_dist=max_dist,
            )
        else:
            framed_missing_mask = None
    else:
        framed_sig = signal.copy()
        if missing_mask is None:
            framed_missing_mask = None
        else:
            framed_missing_mask = missing_mask.copy()
    # Ignore missing values
    if framed_missing_mask is None:
        kernel_mean = float(kernel.mean())
        kernel_std = float(kernel.std())
        # out contains framed_sig mean
        out = xcorr2(framed_sig, kernel1 / kernel_size)

        denom = xcorr2(framed_sig.power(2), kernel1 / kernel_size) - out.power(
            2
        )
        denom = denom.sqrt() * kernel_std
        # quick and dirty hack to robustify: numerical-zeros are turned into
        # real-zeros
        denom.data[np.abs(denom.data) < 1e-10] = 0.0
        denom.data = 1.0 / denom.data

        # store numerator directly in 'out' to avoid multiple replication of
        # data
        out = (
            xcorr2(framed_sig, kernel / kernel_size, tsvd=tsvd)
            - out * kernel_mean
        )
        # Multiply by invert since sparse division is not allowed
        out = out.multiply(denom)

    else:
        # Safety check to make sure mask matches signal
        preproc.check_missing_mask(framed_sig, framed_missing_mask)
        kernel_sum = np.sum(kernel)
        kernel_mean = kernel_sum / kernel_size
        kernel_std = float(kernel.std())
        kernel2_sum = np.sum(kernel ** 2)
        kernel2_mean = kernel2_sum / kernel_size
        # Compute convolution of uniform kernel with missing mask to get number
        # of missing pixels in each window. Will be plugged into Pearson at
        # the end.
        ker1_coo = xcorr2(framed_missing_mask, kernel1).tocoo()
        # From now on, ker1_coo.data contains the number of 'present' samples.
        # (where there is at least one missing pixel)
        ker1_coo.data = kernel_size - ker1_coo.data
        # Compute mean corrected with with number of missing elements (wm)
        kernel_mean_wm = (
            kernel_sum
            - xcorr2(framed_missing_mask, kernel, tsvd=tsvd)[
                ker1_coo.row, ker1_coo.col
            ].A1
        ) / ker1_coo.data
        kernel2_mean_wm = (
            kernel2_sum
            - xcorr2(framed_missing_mask, kernel ** 2, tsvd=tsvd)[
                ker1_coo.row, ker1_coo.col
            ].A1
        ) / ker1_coo.data

        # store signal mean directly in 'out' to avoid multiple replication of
        # data
        out = xcorr2(framed_sig, kernel1 / kernel_size)
        out[ker1_coo.row, ker1_coo.col] = (
            out[ker1_coo.row, ker1_coo.col].A1 * kernel_size / ker1_coo.data
        )

        denom = xcorr2(framed_sig.power(2), kernel1 / kernel_size)
        denom[ker1_coo.row, ker1_coo.col] = (
            denom[ker1_coo.row, ker1_coo.col].A1 * kernel_size / ker1_coo.data
        )

        denom = (denom - out.power(2)) * (kernel2_mean - kernel_mean ** 2)
        denom[ker1_coo.row, ker1_coo.col] = (
            denom[ker1_coo.row, ker1_coo.col].A1
            / (kernel2_mean - kernel_mean ** 2)
            * (kernel2_mean_wm - kernel_mean_wm ** 2)
        )
        denom = denom.sqrt()

        # ensure that enough data points are inside of window
        denom[
            ker1_coo.row[ker1_coo.data < int((1 - missing_tol) * kernel_size)],
            ker1_coo.col[ker1_coo.data < int((1 - missing_tol) * kernel_size)],
        ] = 0.0

        # store numerator directly in 'out' to avoid multiple copies of data
        out *= kernel_mean
        out[ker1_coo.row, ker1_coo.col] = (
            out[ker1_coo.row, ker1_coo.col].A1
            * kernel_mean_wm
            * ker1_coo.data
            / (kernel_mean * kernel_size)
        )
        out = xcorr2(framed_sig, kernel / kernel_size, tsvd=tsvd) - out
        out[ker1_coo.row, ker1_coo.col] = (
            out[ker1_coo.row, ker1_coo.col].A1 * kernel_size / ker1_coo.data
        )

        # Remember where very low values are located in denom to avoid nans
        denom_0 = abs(denom.data) < 1e-10
        # take inverse, because 2 sparse mats can't be divided
        denom.data[~denom_0] = 1 / denom.data[~denom_0]
        denom.data[denom_0] = 0.0
        out = out.multiply(denom)

    # if (max_dist is not None) and sym_upper:
    #    # Trim diagonals further than max_scan_distance
    #    out = preproc.diag_trim(out.tocsr(), max_dist)

    if sym_upper:
        out = sp.triu(out)
    out = out.tocoo()
    out.data[~np.isfinite(out.data)] = 0.0
    out.data[out.data < 0] = 0.0
    out.eliminate_zeros()
    if pval:
        pvals = out.copy()
        if full:
            try:
                # Get number of values for each coeff
                n_obs = ker1_coo.tocsr()[pvals.row, pvals.col].A1
                # Replace implicit n_obs by total kernel size
                n_obs[n_obs == 0] = kernel_size
                pvals.data = cus.corr_to_pval(out.data, n_obs)
            # No nonzero coeff in the matrix, skip calculation of p-values
            except AttributeError:
                pass
        else:
            pvals.data = cus.corr_to_pval(out.data, kernel_size)
        pvals = pvals.tocsr()
        if full:
            pvals = pvals[mk - 1 : -mk + 1, nk - 1 : -nk + 1]
    else:
        pvals = None
    out = out.tocsr()
    if full:
        out = out[mk - 1 : -mk + 1, nk - 1 : -nk + 1]

    return out, pvals


def _normxcorr2_dense(
    signal,
    kernel,
    max_dist=None,
    sym_upper=False,
    full=None,
    missing_mask=None,
    tsvd=None,
    missing_tol=0.75,
    pval=False,
):
    """Computes the normalized cross-correlation of a dense or sparse signal
    and a dense kernel. The resulting matrix contains Pearson correlation
    coefficients.

    Parameters
    ----------
    signal : numpy.array
        The input processed Hi-C matrix.
    kernel : numpy.array or tuple of numpy.arrays
        The pattern kernel to use for convolution. Can be a factorised kernel
        stored in a tuple.
    max_dist : int
        Maximum scan distance, in number of bins from the diagonal. If None,
        the whole matrix is convoluted. Otherwise, pixels further than this
        distance from the diagonal are set to 0 and ignored for performance.
        Only useful for intrachromosomal matrices.
    sym_upper : False
        Whether the matrix is symmetric and upper triangle. True for
        intrachromosomal matrices.
    full : bool
    missing_mask : numpy.array of bools
        Nump
    missing_tol : float
        Proportion of missing values allowed in windows to keep the correlation
        coefficients.
    tsvd : float or None
        If a float between 0 and 1 is given, the input kernel is factorised
        using truncated SVD, keeping enough singular vectors to retain this
        proportion of information. Factorisation speeds up convolution at
        the cost of a loss of information. If the number of singular vectors
        required to retain the desired information isDisabled by default.
    pval : bool
        Whether to return a matrix of p-values.

    Returns
    -------
    numpy.array
        The sparse matrix of correlation coefficients
    """
    mk, nk = kernel.shape
    ms, ns = signal.shape

    # Convert numpy matrices to array to avoid operator overloading
    if isinstance(signal, np.matrix):
        signal = np.array(signal)
    if isinstance(kernel, np.matrix):
        kernel = np.array(kernel)

    if missing_mask is not None:
        return xcorr2(signal, kernel)

    kernel_size = mk * nk
    kernel1 = np.ones(kernel.shape)
    if full:
        framed_sig = np.zeros([ms + 2 * (mk - 1), ns + 2 * (nk - 1)])
        framed_sig[mk - 1 : -mk + 1, nk - 1 : -nk + 1] = signal
        if missing_mask is not None:
            # Add margins around missing mask.
            framed_missing_mask = preproc.frame_missing_mask(
                missing_mask,
                kernel.shape,
                sym_upper=sym_upper,
                max_dist=max_dist,
            )
    else:
        framed_sig = signal
        framed_missing_mask = None
        if missing_mask is not None:
            framed_missing_mask = missing_mask.copy()
    # Pearson correlation
    kernel_mean = float(kernel.mean())
    kernel_std = float(kernel.std())

    kernel1 = np.ones(kernel.shape)
    framed_sig_mean = xcorr2(framed_sig, kernel1 / kernel_size)

    if missing_mask is None:
        out = (
            xcorr2(framed_sig, kernel / kernel_size)
            - framed_sig_mean * kernel_mean
        )
        denom = kernel_std * np.sqrt(
            xcorr2(framed_sig ** 2, kernel1 / kernel_size)
            - framed_sig_mean ** 2
        )
        denom_0 = abs(denom) < 1e-10
        out[~denom_0] /= denom[~denom_0]
        out[denom_0] = 0.0

    else:
        kernel_size = mk * nk - xcorr2(framed_missing_mask, kernel1).toarray()
        kernel_mean = (
            np.sum(kernel) - xcorr2(framed_missing_mask, kernel).toarray()
        ) / kernel_size
        # store signal0_mean in 'out' to avoid multiple replication of data
        out = xcorr2(framed_sig, kernel1) / kernel_size
        denom = xcorr2(framed_sig ** 2, kernel1) / kernel_size
        # Use multiplicative coeff to correct denom by number of missing values
        denom = np.sqrt(
            (denom - out ** 2)
            * (
                (
                    np.sum(kernel ** 2)
                    - xcorr2(framed_missing_mask, kernel ** 2).toarray()
                )
                / kernel_size
                - kernel_mean ** 2
            )
        )
        # ensure that enough data points are inside of window
        denom[kernel_size < int((1 - missing_tol) * mk * nk)] = 0.0

        # store numerator directly in 'out' to avoid multiple copies of data
        out = xcorr2(framed_sig, kernel / kernel_size) - out * kernel_mean
        # Remember where very low values are located in denom to avoid nans
        denom_0 = abs(denom) < 1e-10

        out[~denom_0] /= denom[~denom_0]
        out[denom_0] = 0.0

    if sym_upper:
        out = np.triu(out)
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    if pval:
        if full:
            # Get number of values for each coeff
            n_obs = kernel_size.flatten()
        else:
            n_obs = kernel_size
        pvals = cus.corr_to_pval(out.flatten(), n_obs).reshape(out.shape)
        if full:
            pvals = pvals[mk - 1 : -mk + 1, nk - 1 : -nk + 1]
    else:
        pvals = None
    if full:
        out = out[mk - 1 : -mk + 1, nk - 1 : -nk + 1]
    return out, pvals
