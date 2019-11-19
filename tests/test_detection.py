import numpy as np
import unittest
from nose2.tools import params
import scipy.sparse as sp
import chromosight
import chromosight.utils
import chromosight.utils.detection as cud
import chromosight.utils.io as cio
from matplotlib.mlab import bivariate_normal


mat, chroms, bins, res = cio.load_cool("data_test/example.cool")
mat = mat.tocsr()
#  Get all intra-chromosomal matrices
intra_mats = [
    mat[s:e, s:e] for s, e in zip(chroms["start_bin"], chroms["end_bin"])
]

all_windows = []
all_patterns = []

# Dummy matrices
# Zeros with two isolated points -> speckles
point_mat = np.zeros((10, 10), dtype=float)
# Add isolated 1-pixel points
point_mat[5, 5] = 1
point_mat[2, 2] = 1

# 2D gaussian distribution -> single pattern
X, Y = np.mgrid[-3:3:100j, -3:3:100j]
# 0.1 std, -1 mean on both axes
gauss1 = bivariate_normal(X, Y, 0.1, 0.1, -1.2, -0.5)

# Zeros with 2 separate 2D gaussian distributions -> distinct patterns
gauss2 = bivariate_normal(X, Y, 0.1, 0.1, 0.75, 1.5)
gauss12 = gauss1 + gauss2
gauss12[gauss12 < np.percentile(gauss12, 90)] = 0

# Get all dummy matrices into upper symetric sparse format
upsym = lambda m: sp.coo_matrix(np.triu(m))
point_mat = upsym(point_mat)
gauss1 = upsym(gauss1)
gauss12 = upsym(gauss12)


class TestDetection(unittest.TestCase):
    def test_validate_patterns():
        cud.validate_patterns(
            coords,
            matrix,
            conv_mat,
            detectable_bins,
            kernel_matrix,
            max_undetected_perc,
        )

    def test_pileup_patterns():
        cud.pileup_patterns(all_windows)

    def test_pattern_detector():
        cud.pattern_detector(contact_map, kernel_config, kernel_matrix, area=3)

    def remove_smears():
        cud.remove_smears(patterns, win_size=8)

    def test_picker():
        # Test if speckles are discarded
        assert cud.picker(point_mat, precision=None) is None
        # Test that index is not shifted
        assert cud.picker(gauss1, precision=None)
        # Test number of foci

    def test_label_connected_pixels_sparse():
        cud.label_connected_pixels_sparse(matrix, min_focus_size=2)

    def test_xcorr2():
        cud.xcorr2(signal, kernel, threshold=1e-4)

    def test_corrcoef2d():
        cud.corrcoef2d(
            signal, kernel, max_dist=None, sym_upper=False, scaling="pearson"
        )
