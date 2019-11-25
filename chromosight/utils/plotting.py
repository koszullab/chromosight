import pathlib
import functools
import numpy as np
import chromosight.utils.preprocessing as preproc
from matplotlib import pyplot as plt


def distance_plot(matrices, labels=None, out=None, smooth=True):

    if isinstance(matrices, np.ndarray):
        matrix_list = [matrices]
    else:
        matrix_list = matrices

    if labels is None:
        labels = range(len(matrix_list))
    elif isinstance(labels, str):
        labels = [labels]

    for matrix, name in zip(matrix_list, labels):
        dist = preproc.distance_law(matrix, fun=np.nanmean, smooth=smooth)
        x = np.arange(0, len(dist))
        y = dist
        y[np.isnan(y)] = 0.0
        plt.plot(x, y, label=str(name))
        plt.xlabel("Genomic distance")
        plt.ylabel("Contact frequency")
        # plt.xlim(10 ** 0, 10 ** 3)
        # plt.ylim(10 ** -5, 10 ** -1)
        plt.loglog()
    plt.legend()
    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def pileup_plot(pileup_pattern, name="pileup_patterns", output=None):
    """
    Plot the pileup of all detected patterns
    """
    if output is None:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    plt.imshow(
        pileup_pattern,
        interpolation="none",
        vmin=0.0,
        vmax=2.0,
        cmap="seismic",
    )
    plt.title("{} pileup".format(name))
    plt.colorbar()
    emplacement = output / pathlib.Path(name + ".pdf")
    plt.savefig(emplacement, dpi=100, format="pdf")
    plt.close("all")


def _check_datashader(fun):
    """Decorates function `fun` to check if cooler is available.."""

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        try:
            import datashader

            fun.__globals__["ds"] = datashader
        except ImportError:
            print(
                "The datashader package is required to use {0}, please install it first".format(
                    fun.__name__
                )
            )
            raise
        return fun(*args, **kwargs)

    return wrapped


def plot_whole_matrix(mat, patterns, out=None, region=None, region2=None):
    """
    Visualise the input matrix with a set of patterns overlaid on top.
    Can optionally restrict the visualisation to a region.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The whole genome Hi-C matrix to be visualized.
    patterns : pandas.DataFrame
        The set of patterns to be plotted on top of the matrix. One pattern per
        row, 3 columns: bin1, bin2 and score of types int, int and float, respectively.
    region : tuple of ints
        The range of rows to be plotted in the matrix. If not given, the whole
        matrix is used. It only region is given, but not region2, the matrix is
        subsetted on rows and columns to show a region on the diagonal.
    region2 : tuple of ints
        The range of columns to be plotted in the matrix. Region must also be
        provided, or this will be ignored.

    """
    err_msg = "{var} must be a tuple of indices indicating the range of {dim}."
    if region is not None and region2 is None:
        s1, e1 = region
        s2, e2 = s1, e1
    elif region is not None and region2 is not None:
        s1, e1 = region
        s2, e2 = region2
    else:
        s1, e1 = 0, mat.shape[0]
        s2, e2 = 0, mat.shape[1]

    pat = patterns.copy()
    pat = pat.loc[
        (pat.bin1 > s1) & (pat.bin1 < e1) & (pat.bin2 > s2) & (pat.bin2 < e2),
        :,
    ]
    sub_mat = mat.tocsr()[s1:e1, s2:e2]
    plt.imshow(np.log(sub_mat.todense()), cmap="Reds")
    plt.scatter(
        pat.bin1 - s1, pat.bin2 - s2, facecolors="none", edgecolors="blue"
    )
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
    # cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    # agg = cvs.points(df, "bin1", "bin2", ds.sum("contacts"))
    # img = tf.shade(agg, cmap=["white", "darkred"], how="log")
