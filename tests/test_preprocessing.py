import numpy as np
import pandas as pd
import unittest
from nose2.tools import params
import scipy.stats as ss
import scipy.sparse as sp
import chromosight.utils.preprocessing as preproc
import chromosight.utils.io as cio

mat, chroms, bins, res = cio.load_cool("data_test/example.cool")
mat = mat.tocsr()
# Get all intra-chromosomal matrices
intra_mats = [mat[s:e, s:e] for s, e in zip(chroms.start_bin, chroms.end_bin)]


class TestPreprocessing(unittest.TestCase):
    @params(*intra_mats)
    def test_normalize(self, mat):
        """Check if normalization sets all bins sums to ~1 (0.5 for upper triangle)."""
        norm = preproc.normalize(mat, good_bins=None)
        sym_norm = norm + norm.T - norm.diagonal(0)
        bin_sums = sym_norm.sum(axis=1)
        assert np.all(np.isclose(bin_sums, np.ones(mat.shape[0], rtol=0.1)))

    @params(*intra_mats)
    def test_diag_trim(self, mat):
        """Check if trimming diagonals preserves shape and sets diagonals to zero."""
        for d in range(mat.shape[0]):
            trimmed = preproc.diag_trim(mat, d)
            diag_sums = [trimmed.diagonal(d).sum() for d in range(trimmed.shape[0])]
            assert trimmed.shape == mat.shape
            assert np.sum(diag_sums[d:]) == 0

    @staticmethod
    def despeckle():
        """Check that artificially added outliers are removed"""
        # Make random matrix using uniform sample between 0 and 1
        uniform_mat = sp.random(1000, 1000, 0.1, format="csr")
        # introduce outlier on 11th diag
        uniform_mat[10, 10] = 10e6
        # Compute median of 11th diag (desired value after despeckling)
        exp_val = np.median(uniform_mat.diagonal(10))
        desp = preproc.despeckle(uniform_mat.tocoo(), th2=1)
        assert np.max(uniform_mat.data) > np.max(desp.data)
        assert desp.tocsr()[10, 10] == exp_val

    def test_get_detectable_bins(self):
        """Check if correct bin indices are reported as detectable."""
        # Make random matrix using uniform sample between 0 and 1
        uniform_mat = sp.random(1000, 1000, 0.1, format="csr")
        # introduce outlier bin
        uniform_mat[10, :] = -10e6
        det_bins = preproc.get_detectable_bins(uniform_mat.tocoo(), inter=False)
        # Check if symmetric mode return same detectable rows and cols
        assert np.all(det_bins[0] == det_bins[1])
        # Check if the right bin index is indetectable
        assert 10 not in det_bins[0]
        asym_mat = sp.random(100, 1000, 0.1, format="csr")
        asym[10, :] = -10e6
        asym[:, 6] = -10e6
        det_bins = preproc.get_detectable_bins(self.mat, inter=True)
        assert len(det_bins[0]) < len(det_bins[1])
        # Ensure right index was removed in both dimensions
        assert 10 not in det_bins[0]
        assert 6 in det_bins[0]
        assert 6 not in det_bins[1]
        assert 10 in det_bins[1]
        # Ensure asymmetric intrachromosomal matrix as input results in crash
        with self.assertRaises(ValueError):
            preproc.get_detectable_bins(asym_mat, inter=False)

    @params(*intra_mats)
    def test_detrend(self, mat):
        """Basic test: Check if detrended matrix pixels have lower standard deviation"""
        detrended = preproc.detrend(mat)
        assert mat.data.std() > detrended.data.std()

    @params(*intra_mats)
    def test_ztransform(self, mat):
        """Check if z-transformation yields mean 0 and std 1"""
        ztr = preproc.ztransform(mat)
        assert np.isclose(np.mean(ztr), 0, rtol=0.1)
        assert np.isclose(np.std(ztr), 1, rtol=0.1)

    @staticmethod
    def test_signal_to_noise_threshold():
        """Check if SNR yields correct index on synthetic matrix."""
        # Initialize empty matrix (SNR = 0)
        syn_mat = sp.coo_matrix((1000, 1000))
        # Set all values in 5th first diagonals to 10 (SNR = 10)
        for k in range(5):
            syn_mat.setdiag(10, k=k)
        snr = preproc.signal_to_noise_threshold(mat)
        # Since the 5th first diagonals are good, the last scannable diagonal
        # should be 4
        assert snr == 4

    @params(mat)
    def test_sum_mat_bins(self, mat):
        """Check if bin sum on upper triangle matrix yields expected results."""
        summed = preproc.sum_mat_bins(mat)
        assert np.all(np.isclose(mat.sum(axis=1), summed, rtol=0.1))

    @params(0, 0.1, 0.5, 0.8, 1)
    def test_subsample_contacts_prop(self, prop):
        """Test sampling proportions of contacts"""
        sampled = preproc.subsample_contacts(mat, prop)
        assert np.isclose(sampled.data.sum(), mat.data.sum() * prop, rtol=0.1)

    @params(2, 100, 10000)
    def test_subsample_contacts_count(self, n_contacts):
        """Test sampling raw contact counts"""
        sampled = preproc.subsample_contacts(mat, n_contacts)
        assert np.isclose(sampled.data.sum(), n_contacts, rtol=0.1)

    @params(100000000, 10e10)
    def test_subsample_contacts_exceed(self, n_contacts):
        """Oversampling should result in value errors"""
        with self.assertRaises(ValueError):
            preproc.subsample_contacts(mat, n_contacts)

    @staticmethod
    def test_resize_kernel():
        """
        Ensure resized kernels are of appropriate size and centered and contain
        expected values.
        """
        m = 15
        # Restrict minimum and maximum dimensions of resized kernels
        min_allowed_dim, max_allowed_dim = 5, 101
        # Use a simple point to check if result is centered
        point_kernel = np.zeros((m, m))
        point_kernel[m // 2, m // 2] = 10
        # Try with different combinations of source and target resolutions
        res_list = [3, 900, 10000]
        for kernel_res in res_list:
            for signal_res in res_list:
                exp_dim = int(m * kernel_res / signal_res)
                if not exp_dim % 2:
                    exp_dim -= 1
                obs_kernel = preproc.resize_kernel(
                    point_kernel,
                    kernel_res,
                    signal_res,
                    min_size=min_allowed_dim,
                    max_size=max_allowed_dim,
                )
                obs_dim = obs_kernel.shape[0]
                assert obs_dim == obs_kernel.shape[1]
                assert obs_dim == max(min(max_allowed_dim, exp_dim), min_allowed_dim)
                assert np.max(obs_kernel) == obs_kernel[obs_dim // 2, obs_dim // 2]

