#!/usr/bin/env python3
# coding: utf-8

"""Pattern/Hi-C utils

General purpose utilities related to handling Hi-C contact maps and
loop/border data.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.signal import savgol_filter

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
    ijs : numpy.array of ints
        2D array of coordinates for identified patterns.
    """
    candidate_mat = mat_conv.copy()
    candidate_mat = candidate_mat.tocoo()
    # Compute a threshold from precision arg and set all pixels below to 0
    thres = np.median(mat_conv.data) + precision * np.std(mat_conv.data)
    candidate_mat.data[candidate_mat.data < thres] = 0
    candidate_mat.data[candidate_mat.data != 0] = 1
    candidate_mat.eliminate_zeros()
    # Check if at least one candidate pixel was found
    if len(candidate_mat.data) > 0:
        num_foci, labelled_mat = label_connected_pixels_sparse(candidate_mat)
        # Will hold the coordinates of the best pixel for each focus
        foci_coords = np.zeros([num_foci, 2], int)
        # Iterate over candidate foci
        # NOTE: There can be jumps between foci id due to post labelling
        # removal of single pixel foci in label_connnected_pixels_sparse.
        # This is why we use focus_rank
        for focus_rank, focus_id in enumerate(np.unique(labelled_mat.data)):
            # Remember 1D indices of datapoint in focus
            focus_idx = np.where(labelled_mat.data == focus_id)[0]
            # Find index of max value within those indices
            focus_pixel_idx = np.argmax(mat_conv.data[focus_idx])
            # Retrieve row of original index
            original_pixel_idx = focus_idx[focus_pixel_idx]
            focus_pixel_row = labelled_mat.row[original_pixel_idx]
            focus_pixel_col = labelled_mat.col[original_pixel_idx]
            # Save coords of best pixels for this focus in a table
            foci_coords[focus_rank, 0] = focus_pixel_row
            foci_coords[focus_rank, 1] = focus_pixel_col
    else:
        foci_coords = "NA"
    return foci_coords


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
    candidates : scipy.sparse.coo_matrix:
        The matrix with values replaced by their respective foci
        labels.
    
    Example
    -------
    >>>M.todense()
    [[1 0 0 0]
     [1 0 1 0]
     [1 0 1 1]
     [0 0 0 0]]
    >>>label_foci(M).todense()
    [[1 0 0 0]
     [1 0 2 0]
     [1 0 2 2]
     [0 0 0 0]]
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
    adj_mat = lil_matrix((n_candidates, n_candidates))
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
    num_foci, foci = connected_components(adj_mat)
    foci += 1  # We add 1 so that first spot is not 0
    foci = foci.astype(np.int64)
    # Remove foci with a single pixel
    for focus_num in range(1, num_foci + 1):
        if len(foci[foci == focus_num]) < min_focus_size:
            foci[foci == focus_num] = 0
    # generate a new matrix, similar to the input, but where pixel values
    # are the foci ID of the pixel.
    foci_mat = coo_matrix(
        (foci, (candidates.row, candidates.col)), shape=candidates.shape, dtype=np.int64
    )
    return num_foci, foci_mat


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

    # Removal of speckles (noisy pixels
    clean_mat = despeckles(matrix, 4.0)
    clean_mat = clean_mat.tocsr()

    y = distance_law(clean_mat, detectable_bins[0])
    y[np.isnan(y)] = 0.0
    y_savgol = savgol_filter(y, window_length=17, polyorder=5)

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
    signal, kernel, max_scan_distance=None, threshold=DEFAULT_PRECISION_CORR_THRESHOLD
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
                kernel[ki, kj] * signal[ki : Ms - Mk + 1 + ki, kj : Ns - Nk + 1 + kj]
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
    # Since elementwise sparse matrices division is not implemented, compute
    # numerator and denominator and perform division on the 1D array of nonzero
    # values.
    numerator = corrcoef - mean_signal * mean_kernel
    denominator = std_signal * std_kernel
    corrcoef = numerator.copy()
    # Get coords of non-zero (nz) values in the numerator
    nz_vals = corrcoef.nonzero()
    # Divide them by corresponding entries in the numerator
    denominator = denominator.tocsr()
    corrcoef.data /= denominator[nz_vals].A1
    return corrcoef


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
                # sub_mat = ztransform(sub_mat)
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
