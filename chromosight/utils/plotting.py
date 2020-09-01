"""Chromosight's plotting submodule contains utilities to visualize the pileup
of detected patterns or the input matrix. It also implements an interactive map
recording the coordinates of double clicks."""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt


def pileup_plot(pileup_pattern, output_prefix, name="pileup_patterns"):
    """
    Wrapper around matplotlib.pyplot.imshow to visualize the pileup of patterns
    detected by chromosight
    """

    plt.imshow(
        pileup_pattern,
        interpolation="none",
        vmin=0.0,
        vmax=2.0,
        cmap="seismic",
    )
    plt.title("{} pileup".format(name))
    plt.colorbar()
    plt.savefig(output_prefix + ".pdf", dpi=100, format="pdf")
    plt.close("all")


def plot_whole_matrix(
    clr, patterns, out=None, region=None, region2=None, log_transform=False
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
        row, 3 columns: bin1, bin2 and score of types int, int and float,
        respectively.
    region : str
        The genomic range, in UCSC format, corresponding to rows to be plotted
        in the matrix. If not given, the whole matrix is used. It only region
        is given, but not region2, the matrix is subsetted on rows and columns
        to show a region on the diagonal.
    region2 : str
        The genomic range, in UCSC format, of columns to be plotted in the matrix.
        Region must also be provided, or this will be ignored.
    log_transform : bool
        Whether to log transform the matrix.

    """

    if region is not None and region2 is None:
        mat = clr.matrix().fetch(region)
        s1, e1 = clr.extent(region)
        s2, e2 = s1, e1
    elif region is not None and region2 is not None:
        mat = clr.matrix().fetch(region, region2)
        s1, e1 = clr.extent(region)
        s2, e2 = clr.extent(region2)
    else:
        mat = clr.matrix()[:]
        s1, e1 = 0, clr.shape[0]
        s2, e2 = 0, clr.shape[1]

    pat = patterns.copy()
    pat = pat.loc[
        (pat.bin1 > s1) & (pat.bin1 < e1) & (pat.bin2 > s2) & (pat.bin2 < e2),
        :,
    ]

    if log_transform:
        mat = np.log(mat)
    mat[mat == 0] = np.nan
    plt.figure(dpi=1200)
    plt.imshow(
        mat,
        cmap="Reds",
        vmax=np.percentile(mat[~np.isnan(mat)], 99),
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


def click_finder(mat, half_w=8):
    """
    Given an input Hi-C matrix, show an interactive window and record
    coordinates where the user double-clicks. When the interactive window is
    closed, the stack of windows around recorded coordinates is returned.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The input Hi-C matrix to display interactively.
    half_w : int
        Half width of the windows to return. The resulting windows

    Returns
    -------
    numpy.array :
        3D stack of images around coordinates recorded interactively. The shape
        of the stack is (N, w, w) where N is the number of coordinates and w is
        2*half_w.
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
                print(f"x = {ix}, y = {iy}")
        except IndexError:
            pass
        coords.append((ix, iy))
        return coords

    fig = plt.figure()
    plt.imshow(mat.toarray(), cmap="afmhot_r", vmax=np.percentile(mat.data, 95))
    plt.title("Double click to record pattern positions")
    # Setup click listener
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    # Keep coordinates that were clicked twice consecutively (double-click)
    double_clicked = set()
    for c in range(1, len(coords)):
        if coords[c - 1] == coords[c]:
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


def print_ascii_mat(mat, adjust=True, colored=False, print_str=True):
    """
    Given a 2D numpy array of float, print it in ASCII art.

    Parameters
    ----------
    mat : np.array of floats
        Matrix to visualize.
    adjust : bool
        Whether to adjust the drawing size to termina width.
    colored : bool
        Whether to use colors.
    print_str : bool
        If true, the ASCII art is printed to stdout, otherwise it
        is stored in a string and returned.

    Returns
    -------
    str :
        An empty string is returned if print_str is True, otherwise the
        ASCII art is returned as a string.
    """

    if adjust:
        try:
            term_width = (os.get_terminal_size()[0] // 2) - 5
        except OSError:
            term_width = 79  # default terminal width fallback
        step = int(max(1, np.ceil(mat.shape[1] / term_width)))
    else:
        step = 1
    ascii_str = " .,:;ox%#@"
    ascii_colors = [
        "\x1b[37m",
        "\x1b[37m",
        "\x1b[36m",
        "\x1b[36m",
        "\x1b[32m",
        "\x1b[32m",
        "\x1b[34m",
        "\x1b[34m",
        "\x1b[33m",
        "\x1b[31m",
    ]
    if colored:
        suffix = "\x1b[0m"
    else:
        suffix = ""
    ascii_art = ""

    def ascii_printer(text, end="\n", print_str=print_str):
        if print_str:
            print(text, end=end)
            out = ''
        else:
            out = text + end
        return out

    sorted_pixels = np.sort(mat.flatten())
    perc_pixels = np.searchsorted(sorted_pixels, mat) / len(sorted_pixels)
    perc_pixels = (10 * perc_pixels).astype(int)
    ascii_art += ascii_printer("  " + "- " * (1 + perc_pixels.shape[1] // step))
    for i in range(0, mat.shape[0], step):
        ascii_art += ascii_printer("  |", end="")
        for j in range(0, mat.shape[1], step):  # pixels are skipped
            pix = perc_pixels[i, j]
            prefix = ascii_colors[pix] if colored else ""
            ascii_art += ascii_printer(f"{prefix}{ascii_str[pix]}{suffix} ", end="")
        ascii_art += ascii_printer("|")
    ascii_art += ascii_printer("  " + "- " * (1 + perc_pixels.shape[1] // step))

    return ascii_art
