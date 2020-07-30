# Test suite for chromosight.utils.io
import os
import json
import tempfile
import nose.tools as nt
import numpy as np
import pandas as pd
from jsonschema import ValidationError
import scipy.sparse as sp
import cooler
import chromosight.utils.io as cio

BIN_COLS = ["chrom", "start", "end"]
CHR_COLS = ["name", "length", "start_bin", "end_bin"]


class TestIO:
    def __init__(self):
        """Setup function to generate a named tempfile"""
        # Create tmp temporary file for reading and writing
        tmp_out = tempfile.NamedTemporaryFile(delete=False)
        tmp_out.close()
        self.tmp_path = tmp_out.name

    def rm_tmp(self):
        """Teardown function to remove existing tempfile"""
        # Remove previously created tmp file
        os.unlink(self.tmp_path)

    def test_load_cool(self):
        """Test loading of matrices in cool format"""

        # Write a dummy bedgraph2 (basically a diagonal)
        res, n_bins = 5000, 100000
        chrom_names = ["c1", "c2", "c3"]
        bins_per_chrom = [n_bins // 3, n_bins // 3, n_bins // 3 + n_bins % 3]
        bins = pd.DataFrame(
            {
                "chrom": np.repeat(chrom_names, bins_per_chrom),
                "start": range(0, res * (n_bins), res),
                "end": range(res, res * (n_bins + 1), res),
            }
        )
        pixels = pd.DataFrame(
            {
                "bin1_id": range(n_bins),
                "bin2_id": range(n_bins),
                "count": np.random.randint(0, 100, n_bins),
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
        assert mat.sum() == pixels["count"].sum()

    def test_load_kernel_config(self):
        """
        Check that json config files can be parsed properly and that bad
        configs throw errors.
        """
        # Generate dummy kernel
        m = np.random.random((17, 17))
        kernel_mat_path = self.tmp_path + "kernel"
        # Write kernel matrix to disk
        with open(kernel_mat_path, "w") as kernel:
            np.savetxt(kernel, m)

        # Generate associated config pointing to the kernel
        exp_config = {
            "name": "test_pattern",
            "kernels": [kernel_mat_path],
            "min_dist": 0,
            "max_dist": 10,
            "max_iterations": 1,
            "max_perc_undetected": 10,
            "min_separation": 1,
            "pearson": 0.4,
            "resolution": 1000,
        }
        # Write config to disk
        json.dump(exp_config, open(self.tmp_path, "w"))
        # Load kernel configs and check if values are correct
        obs_config_raw = cio.load_kernel_config(self.tmp_path, custom=True)
        obs_kernel_raw = obs_config_raw["kernels"][0]
        for param in exp_config.keys():
            if param != "kernels":
                assert exp_config[param] == obs_config_raw[param]
        # check if matrix is preserved
        assert np.all(obs_kernel_raw == m)

        # Check if non-existing config yields explicit error
        try:
            cio.load_kernel_config(self.tmp_path + "donotexist", custom=True)
            assert False
        except OSError:
            assert True
        # Check if wrong values in config yields explicit error
        bad_config = exp_config.copy()
        bad_config["max_dist"] = -1
        json.dump(bad_config, open(self.tmp_path, "w"))
        try:
            cio.load_kernel_config(self.tmp_path, custom=True)
            assert False
        except ValidationError:
            assert True
        # Check if missing parameters in config yields explicit error
        bad_config = exp_config.copy()
        bad_config.pop("pearson")
        json.dump(bad_config, open(self.tmp_path, "w"))
        try:
            cio.load_kernel_config(self.tmp_path, custom=True)
            assert False
        except ValidationError:
            assert True
        os.unlink(kernel_mat_path)

    def test_write_patterns(self):
        """Test if pattern coordinates are saved to disk as expected."""
        # Generate dummy pattern list
        res, n_patterns = 5000, 100000
        chrom_names = ["c1", "c2", "c3"]
        bins_per_chrom = [
            n_patterns // 3,
            n_patterns // 3,
            n_patterns // 3 + n_patterns % 3,
        ]
        tmp_coords = pd.DataFrame(
            {
                "chr1": np.repeat(chrom_names, bins_per_chrom),
                "start1": range(0, res * (n_patterns), res),
                "end1": range(res, res * (n_patterns + 1), res),
                "chr2": np.repeat(chrom_names, bins_per_chrom),
                "start2": range(0, res * (n_patterns), res),
                "end2": range(res, res * (n_patterns + 1), res),
                "bin1": range(n_patterns),
                "bin2": range(1, n_patterns + 1),
                "kernel_id": 0,
                "iteration": 0,
                "score": np.random.randint(0, 100, n_patterns),
            }
        )
        for dec in range(1, 5):
            cio.write_patterns(tmp_coords, self.tmp_path, dec=dec)
            obs_coords = pd.read_csv(self.tmp_path + ".tsv", sep="\t")
            assert obs_coords.shape == tmp_coords.shape
            assert np.all(
                np.isclose(obs_coords.score, np.round(tmp_coords.score, dec))
            )
            os.unlink(self.tmp_path + ".tsv")

    def test_save_windows(self):
        """Check that windows around detected patterns can be saved to disk in JSON and npy."""
        tmp_wins = np.random.random((100, 9, 9))
        # Check whether legit windows can be saved and loaded in both formats
        cio.save_windows(tmp_wins, self.tmp_path, fmt="json")
        with open(self.tmp_path + ".json", "r") as jwin:
            w = json.load(jwin)
            # Loaded as a dict, check number of keys
            assert len(w.keys()) == 100
            # Check dim of first value
            assert np.array(w["0"]).shape == (9, 9)
        # Remove json windows file
        os.unlink(self.tmp_path + ".json")

        cio.save_windows(tmp_wins, self.tmp_path, fmt="npy")
        w = np.load(self.tmp_path + ".npy")
        assert w.shape == (100, 9, 9)
        # Remove npy windows file
        os.unlink(self.tmp_path + ".npy")

        # Check if an inappropriate format raises appropriate exception.
        try:
            cio.save_windows(tmp_wins, self.tmp_path, fmt="wrong")
            assert False
        except (ValueError):
            assert True

    def test_check_prefix_dir(self):
        prefix = self.tmp_path
        noexist_prefix = self.tmp_path + "no_exist/prefix"
        # Should work; parent dir is $TMPDIR
        cio.check_prefix_dir(prefix)
        # Should crash: no_exist directory is made up
        try:
            cio.check_prefix_dir(noexist_prefix)
            assert False
        except OSError:
            assert True



