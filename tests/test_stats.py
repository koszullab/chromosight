import numpy as np
import chromosight.utils.stats as cus


def test_fdr_correction():
    """
    Check if BH correction yields same results obtained from R's p.adjust"""
    in_pvals = np.array([0.1, 0.1, 0.05, 0.01])
    exp_qvals = np.array([0.1, 0.1, 0.1, 0.04])
    obs_qvals = cus.fdr_correction(in_pvals)
    assert np.allclose(exp_qvals, obs_qvals)
    assert cus.fdr_correction(None) is None
