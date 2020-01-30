import sys
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


def plot_whole_matrix(
    mat, patterns, out=None, region=None, region2=None, log_transform=False
):
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
    log_transform : bool
        Whether to log transform the matrix.

    """
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
    sub_mat = mat.tocsr()[s1:e1, s2:e2].todense()
    if log_transform:
        sub_mat = np.log(sub_mat)
    sub_mat[sub_mat == 0] = np.nan
    plt.figure(dpi=1200)
    plt.imshow(
        sub_mat,
        cmap="Reds",
        vmax=np.percentile(sub_mat[~np.isnan(sub_mat)], 99),
    )
    plt.scatter(
        pat.bin1 - s1,
        pat.bin2 - s2,
        facecolors="none",
        edgecolors="blue",
        s=0.05,
    )
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
    # cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    # agg = cvs.points(df, "bin1", "bin2", ds.sum("contacts"))
    # img = tf.shade(agg, cmap=["white", "darkred"], how="log")

def click_finder(mat, half_w=8):
    """
    Given an input Hi-C matrix, show an interactive window and record coordinates
    where the user double-clicks. When the interactive window is closed, the stack
    of windows around recorded coordinates is returned.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The input Hi-C matrix to display interactively.
    half_w : int
        Half width of the windows to return. The resulting windows

    Returns
    -------
    numpy.array :
        3D stack of images around coordinates recorded interactively. The shape of
        the stack is (N, w, w) where N is the number of coordinates and w is 2*half_w.
    """
    global coords
    coords = []
    def onclick(event):
        global ix, iy
        global coords
        try:
            ix, iy = int(event.xdata), int(event.ydata)
        except TypeError:
            return None
        try:
            if coords[-1] == (ix, iy):
                print(f'x = {ix}, y = {iy}')
        except IndexError:
            pass
        coords.append((ix, iy))
        return coords

    fig = plt.figure()
    plt.imshow(mat.toarray(), cmap='afmhot_r', vmax = np.percentile(mat.data, 95))
    plt.title('Double click to record pattern positions')
    # Setup click listener
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    # Keep coordinates that were clicked twice consecutively (double-click)
    double_clicked = set()
    for c in range(1, len(coords)):
        if coords[c-1] == coords[c]:
            double_clicked.add(coords[c])
    # initialize empty image stack, will store windows around coords
    img_stack = np.zeros((len(double_clicked), half_w * 2 + 1, half_w * 2 + 1))
    bad_coords = np.zeros(len(double_clicked), dtype=bool)
    # Fill the image stack with windows around each coord
    for i, (center_v, center_h) in enumerate(double_clicked):
            high, low = center_h - half_w, center_h + half_w + 1
            left, right = center_v - half_w, center_v + half_w + 1
            try:
                img_stack[i] = mat[high:low, left:right].toarray()
            except ValueError:
                bad_coords[i] = True
                sys.stderr.write(
                    f"Discarding {(center_v, center_h)}: Too close "
                    "to the edge of the matrix\n"
                )
    # Discard images associated with coords too close to the edge
    img_stack = img_stack[~bad_coords]
    return img_stack


