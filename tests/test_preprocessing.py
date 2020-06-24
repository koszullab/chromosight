import numpy as np
import unittest
from nose2.tools import params
import scipy.sparse as sp
import chromosight
import chromosight.utils
import chromosight.utils.preprocessing as preproc
import chromosight.utils.io as cio

mat, chroms, bins, res = cio.load_cool("data_test/example.cool")
mat = mat.tocsr()
# Get all intra-chromosomal matrices
intra_mats = [
    mat[s:e, s:e] for s, e in zip(chroms["start_bin"], chroms["end_bin"])
]


class TestPreprocessing(unittest.TestCase):
    def test_get_detectable_bins(self):
        """Check if correct bin indices are reported as detectable."""
        # Make random matrix using uniform sample between 0 and 1
        uniform_mat = sp.random(1000, 1000, density=0.1, format="csr")
        # introduce outlier bin
        uniform_mat[10, :] = 0.0
        uniform_mat[:, 10] = 0.0
        uniform_mat = uniform_mat.tocoo()
        uniform_mat.eliminate_zeros()
        det_bins = preproc.get_detectable_bins(
            uniform_mat, inter=False, n_mads=1
        )
        # Check if symmetric mode return same detectable rows and cols
        assert np.all(det_bins[0] == det_bins[1])
        # Check if the right bin index is indetectable
        assert 10 not in det_bins[0]
        asym_mat = sp.random(100, 1000, 0.1, format="csr")
        asym_mat[10, :] = 0
        asym_mat[:, 6] = 0
        asym_mat = asym_mat.tocoo()
        asym_mat.eliminate_zeros()
        asym_mat.data *= 1000
        det_bins = preproc.get_detectable_bins(asym_mat, inter=True, n_mads=1)
        # Ensure correct index was removed in both dimensions
        assert 10 not in det_bins[0]
        assert 6 not in det_bins[1]
        # Ensure asymmetric interchromosomal matrix as input results in crash
        with self.assertRaises(ValueError):
            preproc.get_detectable_bins(asym_mat, inter=False)

    def test_make_missing_mask(self):
        """Test if missing bin masks are generated properly according to matrix type"""
        missing_bins = np.array([0, 4, 9])
        valid_bins = np.array([i for i in range(10) if i not in missing_bins])
        valid_cols = np.array([i for i in range(15) if i not in missing_bins])
        max_dist = 3
        # Symmetric mask, whole matrix masked
        exp_mask_sym = np.zeros((10, 10), dtype=bool)
        exp_mask_sym[:, missing_bins] = True
        exp_mask_sym[missing_bins, :] = True
        # Asymmetric mask, whole matrix masked
        exp_mask_asym = np.zeros((10, 15), dtype=bool)
        exp_mask_asym[:, missing_bins] = True
        exp_mask_asym[missing_bins, :] = True
        # Symmetric mask, only upper triangle masked
        exp_mask_sym_upper = np.triu(exp_mask_sym)
        # Symmetric upper triangle masked up to a certain distance
        exp_mask_sym_upper_maxdist = preproc.diag_trim(
            exp_mask_sym_upper, max_dist + 1
        )
        # Test if correct bins are masked
        obs_mask_sym = preproc.make_missing_mask(
            exp_mask_sym.shape, valid_bins, valid_bins, sym_upper=False
        )
        assert np.all(obs_mask_sym == exp_mask_sym)
        # Test if only upper triangle is masked in upper symmetric matrices
        obs_mask_sym_upper = preproc.make_missing_mask(
            exp_mask_sym.shape, valid_bins, valid_bins, sym_upper=True
        )
        assert np.all(obs_mask_sym_upper == exp_mask_sym_upper)
        # Test masking of asymmetric matrices
        obs_mask_asym = preproc.make_missing_mask(
            exp_mask_asym.shape, valid_bins, valid_cols
        )
        assert np.all(obs_mask_asym == exp_mask_asym)
        # Test if giving an asymmetric matrix with sym_upper results in error
        with self.assertRaises(ValueError):
            preproc.make_missing_mask(
                obs_mask_asym.shape, valid_bins, valid_bins, sym_upper=True
            )
        # Test if using max_dist yields the same results as manually truncating diagonals
        obs_mask_sym_upper_maxdist = preproc.make_missing_mask(
            exp_mask_sym.shape,
            valid_bins,
            valid_bins,
            sym_upper=True,
            max_dist=max_dist,
        )
        assert np.all(obs_mask_sym_upper_maxdist == exp_mask_sym_upper_maxdist)

    @params(100000000, 10e10)
    def test_subsample_contacts_exceed(self, n_contacts):
        """Oversampling should result in value errors"""
        with self.assertRaises(ValueError):
            preproc.subsample_contacts(mat, n_contacts)


@params(*intra_mats)
def test_diag_trim(matrix):
    """Check if trimming diagonals preserves shape and sets diagonals to zero."""
    for d in range(matrix.shape[0]):
        trimmed = preproc.diag_trim(matrix.tocsr(), d)
        diag_sums = [
            trimmed.diagonal(d).sum() for d in range(trimmed.shape[0])
        ]
        assert trimmed.shape == matrix.shape
        assert np.sum(diag_sums[d + 1 :]) == 0


def test_resize_kernel():
    """
    Ensure resized kernels are of appropriate size and centered and contain
    expected values.
    """
    m = 15
    # Restrict minimum and maximum dimensions of resized kernels
    min_allowed_dim = 5
    # Use a simple point to check if result is centered
    point_kernel = np.zeros((m, m))
    point_kernel[m // 2, m // 2] = 10
    # Try with different combinations of source and target resolutions
    res_list = [3, 4, 6, 10]
    for kernel_res in res_list:
        for signal_res in res_list:
            exp_dim = int(m * kernel_res / signal_res)
            if not exp_dim % 2:
                exp_dim -= 1
            obs_kernel = preproc.resize_kernel(
                point_kernel,
                kernel_res=kernel_res,
                signal_res=signal_res,
                min_size=min_allowed_dim,
            )
            obs_kernel_factor = preproc.resize_kernel(
                point_kernel,
                factor=kernel_res / signal_res,
                min_size=min_allowed_dim,
            )

            obs_dim = obs_kernel.shape[0]
            obs_dim_factor = obs_kernel_factor.shape[0]
            assert obs_dim == obs_kernel.shape[1]
            assert obs_dim == obs_dim_factor
            assert obs_dim == max(exp_dim, min_allowed_dim)
            assert np.max(obs_kernel) == obs_kernel[obs_dim // 2, obs_dim // 2]


def test_crop_kernel():
    """
    Ensure cropped kernels are of appropriate size and centered and contain
    expected values.
    """
    m = 15
    # Use a simple point to check if result is centered
    point_kernel = np.zeros((m, m))
    point_kernel[m // 2, m // 2] = 10
    # Try with different combinations of source and target resolutions
    dim_list = range(20)
    for targ in dim_list:
        # Kernel should be made larger to keep odd dimensions
        if targ % 2:
            exp_dim = targ
        else:
            exp_dim = targ + 1
        # If target dimensions is larger than input, cropping is not performed
        if exp_dim > m:
            exp_dim = m
        obs_kernel = preproc.crop_kernel(
            point_kernel, target_size=(targ, targ)
        )
        obs_dim = obs_kernel.shape[0]
        assert obs_dim == exp_dim


def test_zero_pad_sparse():
    """
    Test if zero padding yields correct dimensions and centered input.
    """
    mat = sp.coo_matrix(np.ones((10, 10)))
    for hpad in range(4):
        for vpad in range(4):
            padded = preproc.zero_pad_sparse(mat, margin_h=hpad, margin_v=vpad)
            assert padded.shape[0] == mat.shape[0] + 2 * vpad
            assert padded.shape[1] == mat.shape[1] + 2 * hpad
            assert np.all(
                mat.toarray()
                == padded.toarray()[
                    vpad : padded.shape[0] - vpad,
                    hpad : padded.shape[1] - hpad,
                ]
            )


def test_distance_law():
    """Test if the distance law array has the right dimensions and expected values"""
    m = np.ones((3, 3))
    m += np.array([1, 2, 3])
    # Test regular distance law
    dist_law = preproc.distance_law(sp.csr_matrix(m), smooth=False)
    assert np.all(dist_law == np.array([3.0, 3.5, 4.0]))
    assert dist_law.shape == (3,)
    # Test distance law with isotonic regression: values should not go up
    dist_law = preproc.distance_law(sp.csr_matrix(m), smooth=True)
    assert np.all(dist_law == np.array([3.5, 3.5, 3.5]))
    assert dist_law.shape == (3,)


@params(*intra_mats)
def test_detrend(matrix):
    """Basic test: Check if detrended matrix pixels have lower standard deviation"""
    detrended = preproc.detrend(matrix)
    for d in range(matrix.shape[0] // 10):
        assert matrix.diagonal(d).std() > detrended.diagonal(d).std()


@params(*intra_mats)
def test_ztransform(matrix):
    """Check if z-transformation yields mean 0 and std 1"""
    ztr = preproc.ztransform(matrix)
    assert np.isclose(np.mean(ztr.data), 0, rtol=0.1)
    assert np.isclose(np.std(ztr.data), 1, rtol=0.1)


@params(*intra_mats)
def test_sum_mat_bins(mat):
    """Check if bin sum on upper triangle matrix yields expected results."""
    sym_mat = mat + mat.T
    sym_mat.setdiag(sym_mat.diagonal() / 2)
    summed = preproc.sum_mat_bins(sym_mat)
    exp_sum = 2 * sym_mat.sum(axis=1).A1 - sym_mat.diagonal()
    assert np.all(np.isclose(exp_sum, summed, rtol=0.1))


@params(0, 0.1, 0.5, 0.8, 1)
def test_subsample_contacts_prop(prop):
    """Test sampling proportions of contacts"""
    sampled = preproc.subsample_contacts(
        mat.tocoo(), int(prop * mat.data.sum())
    )
    assert np.isclose(sampled.data.sum(), mat.data.sum() * prop, rtol=0.1)


@params(2, 100, 10000)
def test_subsample_contacts_count(n_contacts):
    """Test sampling raw contact counts"""
    sampled = preproc.subsample_contacts(mat.tocoo(), n_contacts)
    assert np.isclose(sampled.data.sum(), n_contacts, rtol=0.1)

