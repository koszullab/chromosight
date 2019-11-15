# Test suite for chromosight.utils.io
import os
import json
import tempfile
import nose.tools as nt
from nose.tools import with_setup, assert_raises
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cooler
import chromosight.utils.io as cio

BIN_COLS = ['chrom', 'start', 'end', ]
CHR_COLS = ['name', 'length', 'start_bin', 'end_bin']

class TestIO:
    def __init__(self):
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

    def test_load_bedgraph2d(self):
        """Test loading of matrices in bedgraph2 format"""

        # Write a dummy bedgraph2 (basically a diagonal)
        res, n_bins = 5000, 100
        chrom_names = ['c1', 'c2', 'c3']
        bins_per_chrom = [n_bins // 3, n_bins // 3, n_bins // 3 + n_bins % 3]
        # Initialize a dataframe containing the right number of bins
        df = pd.DataFrame(
                {
                    'chrom1': np.repeat(chrom_names, bins_per_chrom),
                    'contacts': np.random.randint(0, 100, n_bins)
                    }
        )
        # Group bins per chromosome
        chr_groups = df.groupby('chrom1')
        # Compute the number of bins in each chromosome and make a range (0 -> nbins)
        start_per_chrom = chr_groups.apply(lambda g:np.array(range(g.shape[0])))
        # Concatenate ranges to have start values from 0 to n bins
        start_array = np.hstack(start_per_chrom)
        # Multiply start values by resolution to get base pair values
        df['start1'] = start_array * res
        # Add binsize to get the end positions
        df['end1'] = df.start1 + res
        # Make the same bins for second pairs (just a diagonal matrix then)
        df[['chrom2', 'start2', 'end2']] = df[['chrom1', 'start1', 'end1']]
        # Reorder columns
        df = df[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'contacts']]
        df.to_csv(self.tmp_path, sep='\t', header=None, index=False)

        # Load bedraph and check whether it was parsed correctly
        mat, chroms, bins, bin_size = cio.load_bedgraph2d(self.tmp_path)
        
        # Median should work to estimate resolution id nbins >> nchroms
        assert res,  np.nanmedian(bins.start.shift(1) - bins.start)
        assert res == bin_size
        assert n_bins == bins.shape[0]
        assert np.all(bins.columns == BIN_COLS)
        assert np.all(chroms.columns == CHR_COLS)
        assert mat.sum() == df.contacts.sum()
                    

    def test_load_cool(self):
        """Test loading of matrices in cool format"""

        # Write a dummy bedgraph2 (basically a diagonal)
        res, n_bins = 5000, 100000
        chrom_names = ['c1', 'c2', 'c3']
        bins_per_chrom = [n_bins // 3, n_bins // 3, n_bins // 3 + n_bins % 3]
        bins = pd.DataFrame(
                {
                    'chrom': np.repeat(chrom_names, bins_per_chrom),
                    'start': range(0, res*(n_bins), res),
                    'end': range(res, res*(n_bins+1), res),
                }
        )
        pixels = pd.DataFrame(
                {
                    'bin1_id': range(n_bins),
                    'bin2_id': range(n_bins),
                    'count': np.random.randint(0, 100, n_bins)
                }
        )

        # Save dataframes into a cool file using cool API
        cooler.create_cooler(self.tmp_path, bins, pixels) 

        # Load cool and check whether it was parsed correctly
        mat, chroms, bins, bin_size = cio.load_cool(self.tmp_path)
        
        # Median should work to estimate resolution id nbins >> nchroms
        assert res == abs(int(np.nanmedian(bins.start.shift(1) - bins.start)))
        assert res == bin_size
        assert n_bins == bins.shape[0]
        assert np.all(bins.columns == BIN_COLS)
        assert np.all(chroms.columns == CHR_COLS)
        assert mat.sum() == pixels['count'].sum()


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
        obs_config_raw = cio.load_kernel_config(self.tmp_path, custom=True)
        obs_kernel_raw = obs_config_raw['kernels'][0]
        for param in exp_config.keys():
            if param != 'kernels':
                assert exp_config[param] == obs_config_raw[param]
        # check if matrix is preserved 
        assert np.all(obs_kernel_raw == m)
        os.unlink(kernel_mat_path)


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
            obs_coords = pd.read_csv(self.tmp_path + ".txt", sep='\t')
            assert obs_coords.shape == tmp_coords.shape
            assert np.all(np.isclose(obs_coords.score, np.round(tmp_coords.score, dec)))
            os.unlink(self.tmp_path + ".txt")


    def test_save_windows(self):
        """Check that windows around detected patterns can be saved to disk in JSON and npy."""
        tmp_wins = np.random.random((100, 9, 9))
        # Check whether legit windows can be saved and loaded in both formats
        cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="json")
        with open(self.tmp_path + ".json", 'r') as jwin:
            w = json.load(jwin)
            # Loaded as a dict, check number of keys
            assert len(w.keys()) == 100
            # Check dim of first value
            assert np.array(w['0']).shape == (9, 9)
        # Remove json windows file
        os.unlink(self.tmp_path + ".json")

        cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="npy")
        w = np.load(self.tmp_path + ".npy")
        assert w.shape == (100, 9, 9)
        # Remove npy windows file
        os.unlink(self.tmp_path + ".npy")

        # Check if an inappropriate format raises appropriate exception.
        with assert_raises(ValueError):
            cio.save_windows(tmp_wins, self.tmp_file, self.tmp_dir, format="wrong")
