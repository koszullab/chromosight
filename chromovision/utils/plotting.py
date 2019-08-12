import pathlib
from matplotlib import pyplot as plt
from . import


def pattern_plot(patterns, matrix, output=None, name=None):
    """Plot patterns

    Plot the matrix with the positions of the found patterns (border or loop)
    on it.
    """
    if name is None:
        name = 0
    if output is None:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    detectable = preprocessing.get_detectable_bins(matrix)
    matscn = preprocessing.normalize(matrix, detectable_bins=detectable)
    try:
        plt.imshow(matscn.toarray() ** 0.15, interpolation="none", cmap="afmhot_r")
    except AttributeError:
        plt.imshow(matscn ** 0.15, interpolation="none", cmap="afmhot_r")
    plt.title(name, fontsize=8)
    plt.colorbar()

    for pattern_type, all_patterns in patterns.items():
        if pattern_type == "borders":
            for border in all_patterns:
                if border[0] != name:
                    continue
                if border[1] != "NA":
                    _, pos1, pos2, _ = border
                    plt.plot(pos1, pos2, "D", color="green", markersize=0.1)
        elif pattern_type == "loops":
            for loop in all_patterns:
                if loop[0] != name:
                    continue
                if loop[1] != "NA":
                    _, pos1, pos2, _ = loop
                    plt.scatter(pos1, pos2, s=15, facecolors="none", edgecolors="blue")

    emplacement = output / pathlib.Path(str(name + 1) + ".pdf2")
    plt.savefig(emplacement, dpi=100, format="pdf")
    plt.close("all")


def distance_plot(matrices, labels=None):

    if isinstance(matrices, np.ndarray):
        matrix_list = [matrices]
    else:
        matrix_list = matrices

    if labels is None:
        labels = range(len(matrix_list))
    elif isinstance(labels, str):
        labels = [labels]

    for matrix, name in zip(matrix_list, labels):
        dist = utils.distance_law(matrix)
        x = np.arange(0, len(dist))
        y = dist
        y[np.isnan(y)] = 0.0
        y_savgol = savgol_filter(y, window_length=17, polyorder=5)
        plt.plot(x, y, "o")
        plt.plot(x)
        plt.plot(x, y_savgol)
        plt.xlabel("Genomic distance")
        plt.ylabel("Contact frequency")
        plt.xlim(10 ** 0, 10 ** 3)
        plt.ylim(10 ** -5, 10 ** -1)
        plt.loglog()
        plt.title(name)
        plt.savefig(pathlib.Path(name) / ".pdf3", dpi=100, format="pdf")
        plt.close("all")


def pileup_plot(pileup_pattern, name="pileup patterns", output=None):
    """
    Plot the pileup of all detected patterns
    """
    if output is None:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    plt.imshow(pileup_pattern, interpolation="none", vmin=0.0, vmax=2.0, cmap="seismic")
    plt.title("pileup {}".format(name))
    plt.colorbar()
    emplacement = output / pathlib.Path(name + ".pdf")
    plt.savefig(emplacement, dpi=100, format="pdf")
    plt.close("all")

