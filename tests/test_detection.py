import numpy as np
import pandas as pd
import unittest
from nose2.tools import params
import scipy.sparse as sp
import scipy.stats as ss
import scipy.signal as sig
import chromosight
import chromosight.utils
import chromosight.utils.detection as cud
import chromosight.utils.io as cio
import chromosight.kernels as ck
from scipy.stats import multivariate_normal


### GENERATING SYNTHETIC DATA
def gauss_mat(meanx, meany, std, shape=(100, 100)):

    # create 2 kernels
    means = (meanx, meany)
    stds = np.eye(2) * std
    k = multivariate_normal(mean=means, cov=stds)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.linspace(-10, 10, shape[0])
    y = np.linspace(-10, 10, shape[1])
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k.pdf(xxyy)

    # reshape and plot image
    g_mat = zz.reshape(shape)
    g_mat = sp.coo_matrix(g_mat)
    return g_mat


# Dummy matrices
# Zeros with two isolated points -> speckles
point_mat = np.zeros((10, 10), dtype=float)
# Add isolated 1-pixel points
point_mat[5, 5] = 0.3
point_mat[2, 2] = 0.3
# Get into upper symmetric sparse format
point_mat = sp.coo_matrix(np.triu(point_mat))

# matrices with 2D gaussian distribution -> single pattern
gauss1_coords = []
gauss1_mats = []
gauss1_windows = []
for mx in np.arange(-1.5, 1.5, 0.5):
    for my in np.arange(-1.5, 1.5, 0.5):
        # Only draw matrices with pattern in upper triangle
        if mx >= my:
            continue
        for sd in np.arange(0.3, 3, 0.3):
            mx, my, sd = [np.round(v, 3) for v in (mx, my, sd)]
            m = gauss_mat(mx, my, sd)
            gauss1_mats.append(m)
            highest = np.where(m.todense() == np.max(m.data))
            c1, c2 = highest[0][0], highest[1][0]
            coords = (c1, c2)
            gauss1_coords.append(coords)
            gauss1_windows.append(
                np.array(
                    gauss1_mats[-1].todense()[
                        c1 - 10 : c1 + 10, c2 - 10 : c2 + 10
                    ]
                )
            )
# Make a 3d stack of windows along axis 0
gauss1_windows = np.swapaxes(np.dstack(gauss1_windows), 0, 2)
# Flip windows into original directions
gauss1_windows = gauss1_windows[:, ::-1, ::-1]
# Zeros with 2 separate 2D gaussian distributions -> distinct patterns
gauss12 = gauss_mat(-6, -2, 0.25, shape=(100, 100))
gauss12 += gauss_mat(1.5, 3, 0.25, shape=(100, 100))
gauss12.data[gauss12.data < np.percentile(gauss12.data, 80)] = 0


# Generate dummy gaussian kernel
gauss_kernel = gauss_mat(0, 0, 5, shape=(7, 7)).todense()
gauss_kernel = gauss_kernel + gauss_kernel.T - np.diag(np.diag(gauss_kernel))


class DummyMap:
    """Simulates ContactMap class, but only includes attributes required for testing"""

    def __init__(
        self, matrix, max_dist=None, detectable_bins=None, inter=False
    ):
        self.matrix = matrix
        self.inter = inter
        self.max_dist = max_dist
        self.detectable_bins = detectable_bins
        if self.detectable_bins is None:
            bins = np.ones(matrix.shape[0], dtype=bool)
            self.detectable_bins = (bins, bins)


@params(*zip(gauss1_mats, gauss1_coords))
def test_validate_patterns(matrix, coords):
    """Test pattern validation"""
    contact_map = DummyMap(matrix)
    conv_mat = cud.normxcorr2(
        matrix, gauss_kernel, max_dist=None, sym_upper=False
    )
    cud.validate_patterns(
        np.array([coords]),
        matrix,
        conv_mat,
        contact_map.detectable_bins,
        gauss_kernel,
        10.0,
    )


def test_pileup_patterns():
    pileup = cud.pileup_patterns(gauss1_windows)
    assert pileup.shape == gauss1_windows.shape[1:]
    # Check if pileup resembles windows
    cor = np.array(
        [
            ss.pearsonr(pileup.flatten(), win.flatten())
            for win in gauss1_windows
        ]
    )
    pearson, pval = cor.mean(axis=0)
    assert pval < 0.05
    assert pearson > 0.5


@params(*gauss1_mats)
def test_pattern_detector(matrix):
    """Test if pattern detector exits correctly"""
    contact_map = DummyMap(matrix, max_dist=100, detectable_bins=None)
    # Gaussian kernel
    kernel_matrix = gauss_mat(0, 0, 5, shape=(7, 7)).todense()
    kernel_config = {
        "max_dist": 100,
        "pearson": 0.5,
        "max_perc_undetected": 10,
    }
    kernel_matrix = (
        kernel_matrix + kernel_matrix.T - np.diag(np.diag(kernel_matrix))
    )
    cud.pattern_detector(contact_map, kernel_config, kernel_matrix)


@params(*range(1, 10))
def test_remove_neighbours(win_size):
    """Check if the correct number of patterns are removed and highest scores are kept"""
    patterns = pd.DataFrame(gauss1_coords, columns=["bin1", "bin2"])
    patterns["bin1"] = patterns["bin1"] // win_size
    patterns["bin2"] = patterns["bin2"] // win_size
    patterns["score"] = np.random.random(patterns.shape[0])
    clean_patterns = cud.remove_neighbours(patterns, win_size=1)
    obs_clean = len(clean_patterns[clean_patterns])
    exp_clean = np.unique(patterns.loc[:, ["bin1", "bin2"]], axis=0).shape[0]
    assert obs_clean == exp_clean
    obs_scores = np.sort(patterns.score[clean_patterns].values)
    exp_scores = np.sort(
        patterns.groupby(["bin1", "bin2"]).max().values.flatten()
    )
    assert np.all(obs_scores == exp_scores)


def test_picker_speckles():
    """Test if speckles are discarded by picker"""
    obs_coords, obs_mat = cud.picker(point_mat, pearson=0.1)
    assert obs_coords is None
    assert obs_mat is None


@params(*zip(gauss1_coords, gauss1_mats))
def test_picker_idx(patterns, matrix):
    """Test that index is not shifted when using picker"""
    thresh = matrix.data.mean()
    obs_coords, _ = cud.picker(matrix, pearson=thresh)
    assert np.all(obs_coords[0] == patterns)


def test_picker_nloci():
    """Test if the number of foci detected by picker is correct"""
    thresh = gauss12.data.mean()
    obs_coords, _ = cud.picker(gauss12, pearson=thresh)
    assert len(obs_coords) == 2


@params(*gauss1_mats)
def test_label_gauss(matrix):
    """Check if correct pixels are members and if no more than 1 focus was found."""
    n_labs, lab_mat = cud.label_foci(matrix)
    assert n_labs == 1
    obs_rows, obs_cols = lab_mat.nonzero()
    exp_rows, exp_cols = matrix.nonzero()
    assert np.all(exp_rows == obs_rows)
    assert np.all(exp_cols == obs_cols)


def test_label_spec():
    """Test label_foci and filter_foci on rectangle matrix with 8-way adjacencies."""

    # Generate a matrix with 4- and 8-way adjacencies to ensure consistent patch
    # labelling.
    label_test_mat = sp.coo_matrix(
        np.array(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
            ]
        )
    )
    exp_labels = sp.coo_matrix(
        np.array(
            [
                [1, 0, 0, 0, 2, 2],
                [1, 0, 3, 0, 0, 0],
                [1, 0, 3, 3, 0, 0],
                [0, 4, 0, 0, 0, 5],
            ]
        )
    )
    n_labs, lab_mat = cud.label_foci(label_test_mat)
    # Check if foci have been separated properly
    assert np.all(exp_labels.data == lab_mat.data)
    assert n_labs == 5
    # Check filtering with min_size >=2 pixels
    num_filt2, _ = cud.filter_foci(lab_mat, min_size=2)
    assert num_filt2 == 3
    # Check filtering with min_size >=3 pixels
    num_filt2, _ = cud.filter_foci(lab_mat, min_size=3)
    assert num_filt2 == 2


@params(*gauss1_mats)
def test_xcorr2(signal):
    """Check if correlation matrix match signal and compare dense vs sparse results"""
    # Get max coordinates of 2D normal in signal
    exp_row, exp_col = np.where(signal.todense() == np.max(signal.todense()))
    # Get max coordinates of correlation scores
    corr_mat_sparse = cud.xcorr2(
        signal, gauss_kernel, threshold=1e-4
    ).todense()
    corr_mat_dense = cud.xcorr2(signal.todense(), gauss_kernel, threshold=1e-4)
    obs_row, obs_col = np.where(corr_mat_dense == np.max(corr_mat_dense))
    # Use scipy result as base truth to compare chromosight results
    corr_mat_scipy = np.zeros(signal.shape)
    kh, kw = (np.array(gauss_kernel.shape) - 1) // 2
    corr_mat_scipy[kh:-kh, kw:-kw] = sig.correlate2d(
        signal.todense(), gauss_kernel, "valid"
    )
    # Apply threshold to scipy result for comparison with xcorr2
    corr_mat_scipy[corr_mat_scipy < 1e-4] = 0
    # Check if best correlation is at the mode of the normal distribution
    # NOTE: There are sometime two maximum values side to side in signal, hence
    # the isin check rather than equality
    assert np.all(np.isin(obs_row, exp_row))
    assert np.all(np.isin(obs_col, exp_col))
    assert np.allclose(
        corr_mat_dense,
        corr_mat_sparse,
        atol=np.mean(corr_mat_dense[corr_mat_dense != 0] / 10),
    )
    assert np.allclose(corr_mat_dense, corr_mat_scipy)


@params(*gauss1_mats)
def test_xcorr2_constant(signal):
    """Check if xcorr2 case for constant kernels yields correct results"""
    k1 = np.ones((11, 11))
    ks = k1.shape[0] * k1.shape[1]
    # convolution of flat kernels in xcorr2 is done by factorising into 2
    # singular vectors and multiplying again to  recover identical results
    # check that this multiplication happens properly in dense and sparse
    # versions
    assert np.allclose(
        cud.xcorr2(signal, k1 / ks).toarray(),
        (cud.xcorr2(signal, k1) / ks).toarray(),
        atol=1e-4,
    )
    assert np.allclose(
        cud.xcorr2(signal.toarray(), k1 / ks),
        cud.xcorr2(signal.toarray(), k1) / ks,
        atol=1e-4,
    )


@params(*gauss1_mats)
def test_normxcorr2(signal):
    """Check if Pearson and cross-product correlations yield appropriate values"""
    corr, pval = cud.normxcorr2(
        signal, gauss_kernel, max_dist=None, sym_upper=False,
    )
    if len(corr.data):
        try:
            assert np.min(corr.data) >= -1
        except AssertionError:
            breakpoint()
        assert np.max(corr.data) <= 1


@params(*gauss1_mats)
def test_normxcorr2_dense_sparse(signal):
    """Check if normxcorr2 yields identical values for dense and sparse versions"""
    corr_d, pval_d = cud.normxcorr2(
        signal.todense(),
        gauss_kernel,
        max_dist=None,
        sym_upper=False,
        pval=True,
    )
    corr_s, pval_s = cud.normxcorr2(
        signal, gauss_kernel, max_dist=None, sym_upper=False, pval=True
    )
    assert np.allclose(corr_s.toarray(), corr_d, rtol=10e-4)
    assert np.allclose(pval_s.toarray(), pval_d, rtol=10e-4)


@params(ck.loops, ck.borders, ck.hairpins)
def test_normxcorr2_kernels(kernel_config):
    """Test corrfoef2d on all built-in patterns"""
    # Loop over the different kernel matrices for the current pattern
    for kernel in kernel_config["kernels"]:
        km, kn = kernel.shape
        # Generate fake Hi-C matrix: empty with pattern centered at 60,80
        pattern_signal = np.zeros((100, 100), dtype=float)
        pattern_signal[
            60 - km // 2 : 60 + (km // 2 + 1),
            80 - kn // 2 : 80 + (kn // 2 + 1),
        ] = kernel
        pattern_signal = sp.csr_matrix(np.triu(pattern_signal))
        # Compute correlation between fake matrix and kernel
        corr, _ = cud.normxcorr2(
            pattern_signal, kernel, max_dist=None, sym_upper=False,
        )

        # Check if the max correlation is where we inserted the pattern
        corr = corr.tocoo()
        corr.data = np.round(corr.data, 3)
        obs_row = corr.row[np.where(corr.data == np.max(corr.data))]
        obs_col = corr.col[np.where(corr.data == np.max(corr.data))]
        assert 60 in obs_row
        assert 80 in obs_col


# TODO: Add tests for inter (asymmetric) matrices
# TODO: Test for error handling
