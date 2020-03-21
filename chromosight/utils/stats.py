import numpy as np
import scipy.stats as ss


def fdr_correction(pvals):
    """
    Applies false discovery rate correction via the
    Benjamini-Hochberg procedure to adjust input p-values
    for multiple testing.

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
        fdr = None
    else:
        ranked_pvals = ss.rankdata(pvals)
        fdr = pvals * len(pvals) / ranked_pvals
        fdr[fdr > 1] = 1

    return fdr


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
    n = np.repeat(n, corr.shape)

    means = np.arctanh(corr)
    se = 1 / np.sqrt(n - 3)
    z = (corr - means) / se
    # Get values of the cumulative standard distribution for each zscore
    # Zscores are all set to negative so as to obtain the tail, resulting
    # p-values are multiplied by two to obtain the two-sided values
    # (distribution is symmetric)
    pvals = 2 * ss.norm().cdf(-np.abs(z))
    return np.log10(pvals)
