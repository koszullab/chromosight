import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import chromosight.utils.preprocessing as preproc


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
