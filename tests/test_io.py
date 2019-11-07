# Test suite for chromosight.utils.io
import os
import json
import tempfile
import nose.tools as nt
from nose.tools import with_setup, assert_raises
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss
import cooler
import chromosight.utils.io as cio

BIN_COLS = ['chrom', 'start', 'end', ]
CHR_COLS = ['name', 'length', 'start_bin', 'end_bin']

class TestIO:
    def setup(self):
        """Setup function to generate a named tempfile"""
        # Create tmp temporary file for reading and writing
        tmp_out = tempfile.NamedTemporaryFile(delete=False)
        tmp_out.close()
        # Give access to full path, dirname and basename in diff variables
        self.tmp_path = tmp_out.name
        self.tmp_dir = os.path.dirname(self.tmp_path)
        self.tmp_file = os.path.basename(self.tmp_path)

    def rm_tmp(self):
        """Teardown function to remove existing tempfile"""
        # Remove previously created tmp file
        os.unlink(self.tmp_path)

    #@with_setup(gen_tmp, rm_tmp)
    def test_load_bedgraph2d(self):
        """Test loading of matrices in bedgraph2 format"""

        # Write a dummy bedgraph2 (basically a diagonal)
        res, n_bins = 5000, 100000
        chrom_names = ['c1', 'c2', 'c3']
        bins_per_chrom = [n_bins // 3, n_bins // 3, n_bins // 3 + n_bins % 3]
        df = pd.DataFrame(
                {
                    'chrom1': np.repeat(chrom_names, bins_per_chrom),
                    'start1': range(0, res*(n_bins), res),
                    'end1': range(res, res*(n_bins+1), res),
                    'chrom2': np.repeat(chrom_names, bins_per_chrom),
                    'start2': range(0, res*(n_bins), res),
                    'end2': range(res, res*(n_bins+1), res),
                    'contacts': np.random.randint(0, 100, n_bins)
                    }
        )
        df.to_csv(tmp_path, sep='\t', header=None, index=False)

        # Load bedraph and check whether it was parsed correctly
        mat, chroms, bins, bin_size = cio.load_bedgraph2d(tmp_path)
        
        # Median should work to estimate resolution id nbins >> nchroms
        assert res == np.nanmedian(bins.start.shift(1) - bins.start)
        assert res == bin_size
        assert n_bins == bins.shape[0]
        assert bins.columns == BIN_COLS
        assert chroms.columns == CHR_COLS
        assert mat.sum() == df.contacts.sum()
                    

    #@with_setup(gen_tmp, rm_tmp)
    def test_load_cool(self):
        """Test loading of matrices in cool format"""

        # Write a dummy bedgraph2 (basically a diagonal)
        res, n_bins = 5000, 100000
        chrom_names = ['c1', 'c2', 'c3']
        bins_per_chrom = [n_bins // 3, n_bins // 3, n_bins // 3 + n_bins % 3]
        df = pd.DataFrame(
                {
                    'chr1': np.repeat(chrom_names, bins_per_chrom),
                    'start1': range(0, res*(n_bins), res),
                    'end1': range(res, res*(n_bins+1), res),
                    'chr2': np.repeat(chrom_names, bins_per_chrom),
                    'start2': range(0, res*(n_bins), res),
                    'end2': range(res, res*(n_bins+1), res),
                    'contacts': np.random.randint(0, 100, n_bins)
                    }
        )
        df.to_csv(tmp_path, sep='\t', header=None, index=False)

        # Load bedraph and check whether it was parsed correctly
        mat, chroms, bins, bin_size = cio.load_cool(tmp_path)
        
        # Median should work to estimate resolution id nbins >> nchroms
        assert res == np.nanmedian(bins.start.shift(1) - bins.start)
        assert res == bin_size
        assert n_bins == bins.shape[0]
        assert bins.columns == BIN_COLS
        assert chroms.columns == CHR_COLS
        assert mat.sum() == df.contacts.sum()


    #@with_setup(gen_tmp, rm_tmp)
    def test_load_kernel_config(self):
        """Check that json config files can be parsed properly"""
        # Generate dummy kernel
        m = np.random.random((17, 17))
        kernel_mat_path = self.tmp_path + "kernel"
        # Write kernel matrix to disk
        with open(kernel_mat_path, 'w') as kernel:
            np.savetxt(kernel, m)
        
        # Generate associated config pointing to the kernel
        exp_config = {
            "name": "test_pattern",
            "kernels": [
               kernel_mat_path 
            ],
            "max_dist": 10,
            "max_iterations": 1,
            "max_perc_undetected": 10,
            "precision": 4,
            "resolution": 1000,
        }
        # Write config to disk
        with open(self.tmp_path, 'w') as f:
            json.dump(exp_config, f)
        # Load kernel configs and check if values are correct 
        obs_config_raw = cio.load_kernel_config(self.tmp_path, custom=False, zscore=False)
        obs_kernel_raw = obs_config_raw['kernels'][0]
        for param in exp_config.keys():
            if param != 'kernels':
                assert exp_config[param] == obs_config_raw[param]

        # check if matrix is preserved with and without zscore transformation
        assert obs_kernel_raw == m

        obs_config_zscore = cio.load_kernel_config(tmp_config, custom=False, zscore=False)
        obs_kernel_zscore = obs_config_zscore['kernels'][0]
        assert obs_kernel_zscore == ss.zscore(m, axis=None)
        os.unlink(kernel_mat_path)



    def test_dense2sparse(self):
        """Check results of dense to sparse matrix converter"""
        # Generate a random sparse matrix (with fixed seed)
        s = sp.random(1000, 1000, density=0.01, format='coo', dtype=np.float, random_state=1)
        # Make corresponding dense mat and get upper triangle
        m = s.todense()
        s_exp = coo_matrix(np.triu(m))
        # Check if function returns the same data as computed on dense matrix for each format
        for fmt in ['coo', 'csr', 'csc', 'lil']:
            s_obs = cio.dense2sparse(m, format=fmt)
            assert m_exp == m_obs.tocoo()


    #@with_setup(gen_tmp, rm_tmp)
    def test_write_patterns(self):
        """Test if pattern coordinates are saved to disk as expected."""
        # Generate dummy pattern list
        res, n_patterns = 5000, 100000
        chrom_names = ['c1', 'c2', 'c3']
        bins_per_chrom = [n_patterns// 3, n_patterns // 3, n_patterns // 3 + n_patterns % 3]
        tmp_coords = pd.DataFrame(
                {
                    'chr1': np.repeat(chrom_names, bins_per_chrom),
                    'start1': range(0, res*(n_patterns), res),
                    'end1': range(res, res*(n_patterns+1), res),
                    'chr2': np.repeat(chrom_names, bins_per_chrom),
                    'start2': range(0, res*(n_patterns), res),
                    'end2': range(res, res*(n_patterns+1), res),
                    'bin1': range(n_patterns),
                    'bin2': range(1, n_patterns+1),
                    'kernel_id': 0,
                    'iteration': 0,
                    'score': np.random.randint(0, 100, n_patterns)
                    }
        )
        for dec in range(1,5):
            cio.write_patterns(tmp_coords, self.tmp_file, self.tmp_dir, dec=dec)
            obs_coords = pd.read_csv(self.tmp_path, sep='\t')
            assert obs_coords.shape == tmp_coords.shape
            assert np.all(np.close(obs_coords.score, np.round(tmp_coords.score, dec)))


    #@with_setup(gen_tmp, rm_tmp)
    def test_save_windows(self):
        tmp_wins = np.random.random((100, 9, 9))
        # Check whether legit windows can be saved and loaded in both formats
        cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="json")
        with open(self.tmp_path + ".json", 'r') as jwin:
            w = json.load(jwin)
            assert w.shape == (100, 9, 9)

        cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="npy")
        with open(self.tmp_path + ".npy", 'r') as nwin:
            w = np.load(nwin)
            assert w.shape == (100, 9, 9)

        # Check if an inappropriate format raises appropriate exception.
        with assert_raises(ValueError):
            cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="wrong")
