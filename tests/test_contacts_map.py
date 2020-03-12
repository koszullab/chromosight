import numpy as np
import pandas as pd
import tempfile
import pathlib
import unittest
from nose2.tools import params
from os.path import isfile, join
import scipy.sparse as sp
import chromosight.utils.io as cio
import chromosight.utils.contacts_map as ccm
import chromosight.kernels as ck
import chromosight.utils.preprocessing as preproc

COOL_TEST = "data_test/example.cool"
#BG2_TEST = "data_test/example.bg2"


class DummyGenome:
    def __init__(self):
        ...

    @ccm.DumpMatrix(None)
    def nodump_method(self):
        return False

    @ccm.DumpMatrix("test_dump")
    def dump_method(self):
        return True


def test_dump_matrix():
    """Decorate dummy classes with DumpMatrix to check expected behaviour"""
    # Just checking method execution
    dummy_genome = DummyGenome()
    assert dummy_genome.dump_method() == True
    assert dummy_genome.nodump_method() == False
    # Actual dumping
    dummy_genome.matrix = sp.random(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_genome.dump = tmpdir
        # Without instance name
        dummy_genome.dump_method()
        assert isfile(join(tmpdir, "test_dump.npz"))
        # With instance name
        dummy_genome.name = "test_instance"
        dummy_genome.dump_method()
        assert isfile(join(tmpdir, "test_instance_test_dump.npz"))


@params(COOL_TEST)
def test_hic_genome(path):
    """Test HicGenome instantiation and matrix splitting with bg2 and cool files"""
    # Simple instantiation test: no kernel config, no inter matrices
    hic_genome = ccm.HicGenome(path)
    hic_genome.make_sub_matrices()
    n_chroms = len(hic_genome.clr.chromnames)
    assert hic_genome.sub_mats.shape[0] == n_chroms
    assert hic_genome.max_dist is None

    # Test with inter + kernel
    hic_genome = ccm.HicGenome(path, inter=True, kernel_config=ck.loops)
    hic_genome.make_sub_matrices()
    assert hic_genome.sub_mats.shape[0] == n_chroms ** 2 - n_chroms
    assert hic_genome.max_dist == ck.loops["max_dist"] // hic_genome.clr.binsize
    assert hic_genome.largest_kernel == ck.loops["kernels"][0].shape[0]


@params(COOL_TEST)
def test_hic_genome_normalize(path):
    """Test if normalization of HicGenome object yields expected results"""
    hic_genome = ccm.HicGenome(path)
    hic_genome.normalize(force_norm=True)


class TestHicGenome(unittest.TestCase):
    @params(COOL_TEST)
    def test_hic_genome_subsample(self, path):
        """Check results and error handling of contacts subsampling"""
        with self.assertRaises(ValueError):
            ccm.HicGenome(path, sample=-1)
        with self.assertRaises(ValueError):
            ccm.HicGenome(path, sample="a")
        hic_genome = ccm.HicGenome(path)
        hic_genome.make_sub_matrices()
        for sub in hic_genome.sub_mats.contact_map:
            (s1, e1), (s2, e2) = sub.extent
            sub.matrix = sub.clr.matrix(sparse=True, balance=True)[s1:e1, s2:e2]
            sub.matrix.data[np.isnan(sub.matrix.data)] = 0
            sub.matrix.eliminate_zeros()
            ori_sum = sub.matrix.sum()
            sub.subsample(0.7)
            print(0.7 * ori_sum, sub.matrix.sum())
            assert np.isclose(int(0.7 * ori_sum), sub.matrix.sum())


@params(COOL_TEST)
def test_hic_genome_get_full_mat_pattern(path):
    """Test sub matrix to full matrix bin conversion"""
    hic_genome = ccm.HicGenome(path)
    dummy_patterns = pd.DataFrame({"bin1": [0, 10, 50], "bin2": [1, 11, 51]})
    obs_coords = hic_genome.get_full_mat_pattern(
        "chr1", "chr1", dummy_patterns
    )
    assert np.all(obs_coords.bin1 == dummy_patterns.bin1)
    assert np.all(obs_coords.bin2 == dummy_patterns.bin2)


@params(COOL_TEST)
def test_hic_genome_get_sub_mat_pattern(path):
    """Test full matrix to sub matrix bin conversion"""
    hic_genome = ccm.HicGenome(path)
    dummy_patterns = pd.DataFrame({"bin1": [0, 10, 50], "bin2": [1, 11, 51]})
    obs_coords = hic_genome.get_sub_mat_pattern("chr1", "chr1", dummy_patterns)
    assert np.all(obs_coords.bin1 == dummy_patterns.bin1)
    assert np.all(obs_coords.bin2 == dummy_patterns.bin2)


@params(COOL_TEST)
def test_hic_genome_bins_to_coords(path):
    """Test conversion of bins to genomic coordinates"""
    hic_genome = ccm.HicGenome(path)
    idx = [0, 5, 8]
    exp_bins = hic_genome.bins.iloc[idx, :]
    obs_bins = hic_genome.bins_to_coords(idx)
    assert np.all(exp_bins == obs_bins)


@params(COOL_TEST)
def test_hic_genome_coords_to_bins(path):
    """Test conversion of bins to genomic coordinates"""
    hic_genome = ccm.HicGenome(path)
    coords = pd.DataFrame({"chrom": ["chr1", "chr2"], "pos": [150, 4000]})
    exp_bins = np.array([0, 131])
    obs_bins = hic_genome.coords_to_bins(coords)
    assert np.all(exp_bins == obs_bins)


#@params(*zip((50, 100, 500), (5, 10, 100), (10, 20, 50)))
#def test_contact_map_intra(size, max_dist, kernel_size):
#    """Test preprocessing for intra matrices"""
#    rand_mat = sp.random(size, size, density=0.95)
#    contact_map = ccm.ContactMap(rand_mat)
#    # Check if detrending yields average values of 1 (in upper triangle)
#    assert np.isclose(
#        np.mean(preproc.sum_mat_bins(contact_map.matrix)) / size, 1, rtol=0.1
#    )
#
#    contact_map = ccm.ContactMap(
#        rand_mat, max_dist=max_dist, largest_kernel=kernel_size
#    )
#    diagomeans = np.array(
#        [np.mean(d[d != 0]) for d in contact_map.matrix.todia().data]
#    )
#    diagoffsets = contact_map.matrix.todia().offsets
#    # Check if matrix is trimmed after max dist + largest kernel
#    assert max(diagoffsets) == max_dist + kernel_size
#    # OBSOLETE: Check if a margin of ones was inserted below diagonal
#    # assert np.allclose(np.mean(diagomeans[np.where(diagoffsets < 0)[0]]), 1)
#
#
#def test_contact_map_inter():
#    rand_mat = sp.random(100, 500, density=0.95)
#    contact_map = ccm.ContactMap(rand_mat, inter=True)
#    assert np.isclose(np.mean(contact_map.matrix), 1, rtol=0.1)
