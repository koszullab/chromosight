import numpy as np
import scipy.stats as ss


def fdr_correction(pvals):
    """
    Applies false discovery rate correction via the
    Benjamini-Hochberg procedure to adjust input p-values
    for multiple testing. .

    Parameters
    ----------
    pvals : numpy.array of floats
        Array of uncorrected p-values.

    Returns
    -------
    fdr : numpy.array of floats
        Array of corrected p-values (q-values).
    """
    if pvals is None:
        qvals = None
    else:
        pvals = np.array(pvals)
        # get descending indices
        by_descend = pvals.argsort()[::-1]
        # Get mapping from sorted to original indices
        by_orig = by_descend.argsort()
        # Compute the ratio of n_pval by ranks, which can then be
        # multiplied by p-values to get the critical value
        steps = float(len(pvals)) / np.arange(len(pvals), 0, -1)
        # Compute critical values and find last p-value below above it
        # Set all other to the same value and back to 1 if higher
        qvals = np.minimum(1, np.minimum.accumulate(steps * pvals[by_descend]))
        # Unsort q-values to get original order
        qvals = qvals[by_orig]

    return qvals


def corr_to_pval(corr, n, n_missing=None):
    """
    Given a list of Pearson correlation coefficient,
    convert them to two-sided log10 p-values. The p-values
    are computed via the fisher transformation described
    on: https://w.wiki/Ksu

    Parameters
    ----------
    corr : numpy.array
        The Pearson correlation coefficients.
    n : int or numpy.array
        The number of observations used to compute correlation
        coefficients. Can be given as an array of the same size as corr
        to give the number of sample in each coefficient.

    Returns
    -------
    numpy.array
        The array of log10-transformed tw-sided p-values,
        same size as corr.
    """
    if isinstance(n, int):
        n = np.repeat(n, corr.shape)
    elif isinstance(n, np.ndarray):
        if n.shape != corr.shape:
            raise ValueError("corr and n must have identical shapes.")
    # Apply Fisher z-transformation on coefficients
    z1 = 1.1513 * np.log10((1 + corr) / (1 - corr))
    se = 1 / np.sqrt(n - 3)
    z_score = z1 / se
    # Get values of the cumulative standard distribution for each zscore
    # Zscores are all set to negative so as to obtain the tail, resulting
    # p-values are multiplied by two to obtain the two-sided values
    # (distribution is symmetric)
    pvals = 2 * ss.norm().cdf(-np.abs(z_score))

    return np.log10(pvals)
