from nose2.tools import params
import tempfile
import os
import numpy as np
import pandas as pd
import chromosight.utils.plotting as cup
import chromosight.utils.detection as cud
import chromosight.utils.io as cio


mat, chroms, bins, res = cio.load_cool("data_test/example.cool")

mat = mat.tocsr()
#  Get all intra-chromosomal matrices
intra_mats = [
    mat[s:e, s:e] for s, e in zip(chroms["start_bin"], chroms["end_bin"])
]

pattern_list = []
window_list = []


class TestPlotting:
    def __init__(self):
        """Setup function to generate a named tempfile"""
        # Create tmp temporary file for reading and writing
        tmp_out = tempfile.NamedTemporaryFile(delete=False)
        tmp_out.close()
        # Give access to full path, dirname and basename in diff variables
        self.tmp_path = tmp_out.name

    def test_distance_plot(self):
        cup.distance_plot(intra_mats, labels=None, out=self.tmp_path)

    def test_pileup_plot(self):
        windows = np.reshape(np.random.randint(100, size=1000), (10, 10, 10))
        pileup_pattern = cud.pileup_patterns(windows)
        cup.pileup_plot(pileup_pattern, self.tmp_path, name="pileup_patterns")

    @params(*intra_mats)
    def test_distance_plot(self, matrix):
        rand_rows = np.random.randint(0, matrix.shape[0], size=10)
        rand_cols = np.random.randint(0, matrix.shape[1], size=10)
        patterns = pd.DataFrame({"bin1": rand_rows, "bin2": rand_cols})
        cup.plot_whole_matrix(matrix, patterns, out=self.tmp_path)

