import numpy as np
import pandas as pd
import unittest
from nose2.tools import params
import scipy.sparse as sp
import scipy.stats as ss
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
point_mat[5, 5] = 1
point_mat[2, 2] = 1
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

    def __init__(self, matrix, max_dist=None, detectable_bins=None):
        self.matrix = matrix
        self.max_dist = max_dist
        self.detectable_bins = detectable_bins
        if self.detectable_bins is None:
            bins = np.ones(matrix.shape[0], dtype=bool)
            self.detectable_bins = (bins, bins)


@params(*zip(gauss1_mats, gauss1_coords))
def test_validate_patterns(matrix, coords):
    """Test pattern validation"""
    contact_map = DummyMap(matrix)
    conv_mat = cud.corrcoef2d(
        matrix, gauss_kernel, max_dist=None, sym_upper=False, scaling="pearson"
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
        "precision": 1,
        "max_perc_undetected": 10,
    }
    kernel_matrix = (
        kernel_matrix + kernel_matrix.T - np.diag(np.diag(kernel_matrix))
    )
    cud.pattern_detector(contact_map, kernel_config, kernel_matrix)


@params(*range(1, 10))
def test_remove_smears(win_size):
    """Check if the correct number of patterns are removed and highest scores are kept"""
    patterns = pd.DataFrame(gauss1_coords, columns=["bin1", "bin2"])
    patterns["bin1"] = patterns["bin1"] // win_size
    patterns["bin2"] = patterns["bin2"] // win_size
    patterns["score"] = np.random.random(patterns.shape[0])
    clean_patterns = cud.remove_smears(patterns, win_size=1)
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
    assert cud.picker(point_mat, precision=None) is None


@params(*zip(gauss1_coords, gauss1_mats))
def test_picker_idx(patterns, matrix):
    """Test that index is not shifted when using picker"""
    obs_coord = cud.picker(matrix, precision=None)[0]
    assert np.all(obs_coord == patterns)


def test_picker_nloci():
    """Test if the number of foci detected by picker is correct"""
    assert len(cud.picker(gauss12, precision=1)) == 2


@params(*gauss1_mats)
def test_label_connected_pixels_sparse(matrix):
    """Check if correct pixels are members and if no more than 1 focus was found."""
    n_labs, lab_mat = cud.label_connected_pixels_sparse(
        matrix, min_focus_size=2
    )
    assert n_labs == 1
    obs_rows, obs_cols = lab_mat.nonzero()
    exp_rows, exp_cols = matrix.nonzero()
    assert np.all(exp_rows == obs_rows)
    assert np.all(exp_cols == obs_cols)


@params(*gauss1_mats)
def test_xcorr2(signal):
    """Check if correlation matrix match signal"""
    # Get max coordinates of 2D normal in signal
    exp_row, exp_col = np.where(signal.todense() == np.max(signal.todense()))
    # Get max coordinates of correlation scores
    corr_mat = cud.xcorr2(signal, gauss_kernel, threshold=1e-4).todense()
    obs_row, obs_col = np.where(corr_mat == np.max(corr_mat))
    # Check if best correlation is at the mode of the normal distribution
    # NOTE: There are sometime two maximum values side to side in signal, hence
    # the isin check rather than equality
    assert np.all(np.isin(obs_row, exp_row))
    assert np.all(np.isin(obs_col, exp_col))


@params(*gauss1_mats)
def test_corrcoef2d(signal):
    """Check if Pearson and cross-product correlations yield appropriate values"""
    for scaling in ["pearson", "cross"]:
        corr = cud.corrcoef2d(
            signal,
            gauss_kernel,
            max_dist=None,
            sym_upper=False,
            scaling=scaling,
        )
        if len(corr.data):
            assert np.min(corr.data) >= 0
            assert np.max(corr.data) <= 1


@params(ck.loops, ck.borders, ck.hairpins)
def test_corrcoef2d_kernels(kernel_config):
    """Test corrfoef2d on all built-in patterns"""
    # Loop over the different kernel matrices for the current pattern
    for kernel in kernel_config["kernels"]:
        for scaling in ["pearson", "cross"]:
            km, kn = kernel.shape
            # Generate fake Hi-C matrix: empty with pattern centered at 60,80
            pattern_signal = np.zeros((100, 100), dtype=float)
            pattern_signal[
                60 - km // 2 : 60 + (km // 2 + 1),
                80 - kn // 2 : 80 + (kn // 2 + 1),
            ] = kernel
            pattern_signal = sp.coo_matrix(np.triu(pattern_signal))
            # Compute correlation between fake matrix and kernel
            corr = cud.corrcoef2d(
                pattern_signal,
                kernel,
                max_dist=None,
                sym_upper=False,
                scaling=scaling,
            )

            # Check if the max correlation is where we inserted the pattern
            corr = corr.tocoo()
            obs_row = corr.row[np.where(corr.data == np.max(corr.data))]
            obs_col = corr.col[np.where(corr.data == np.max(corr.data))]
            assert obs_row == 60
            assert obs_col == 80


# TODO: Add tests for inter (asymmetric) matrices
# TODO: Test for error handling
