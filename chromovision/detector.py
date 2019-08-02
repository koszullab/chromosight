#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromovision detect <contact_map> [<output>] [--kernels=None] [--loops]
                        [--borders] [--precision=4] [--iterations=auto]
                        [--inter] [--input-format cool]

Arguments:
    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    contact_map                The Hi-C contact map to detect patterns on, in
                                bedgraph2d or cool format. 
    output                      name of the output directory
    -k None, kernels None       A custom kernel template to use, if not using
                                one of the presets. If not supplied, the
                                loops or borders option must be used.
                                [default: None]
    -L, --loops                 Whether to detect chromatin loops.
    -B, --borders               Whether to detect domain borders.
    -p 4, --precision 4         Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: 4]
    -I, --inter                 Use to consider interchromosomal contacts.
    -i auto, --iterations auto  How many iterations to perform after the first
                                template-based pass. Auto means iterations are
                                performed until convergence. [default: auto]
    -f cool, --input-format cool Input format of the contact map. Can be csv, bg2
                                or cool. [default: cool]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pathlib
import os
import functools
import docopt
from chromovision.version import __version__
from chromovision.contacts_map import ContactMap
from chromovision import utils
from chromovision import io
from scipy import sparse

MAX_ITERATIONS = 3


def pattern_detector(
    contact_map,
    kernel,
    pattern_type="loops",
    precision=4.0,
    area=8,
    undetermined_percentage=1.0,
    nb_patterns=[],
):
    """Pattern detector

    Detect patterns by iterated kernel matching, and compute the resulting
    'agglomerated pattern' as matched on the matrices.

    Parameters
    ----------
    contact_map : ContactMap object
        An object containing a Hi-C contact map with chromosome indices, 
        chromosome names,  inter and intra chromosome sub-matrices and other
        attributes.
    kernel : array_like
        The initial template for pattern matching in the first pass.
    pattern_type : str, optional
        If set to "borders" or "loops", filtering is performed in order to
        remove spurious false positives (such as far-off loops or off-diagonal
        borders). Default is "loops".
    precision : float, optional
        Controls the amount of false positives. A higher precision means fewer
        detected patterns overall and fewer false positives. Default is 4.0.
    area : int, optional
        The window size of the agglomerated pattern. The final windows will have
        a width of 2 * area + 1. Default is 8.
    undetermined_percentage : float, optional
        How much missing data is tolerated in the pattern windows. Patterns
        with a percentage area above this parameter with only missing data are
        discarded. Default is 1., i.e. one percent.
    Returns
    -------
    detected_pattern : list
        A list of detected patterns in tuple form: (name, x, y, score).
    agglomerated_pattern : np.ndarray
        The 'agglomerated' (element-wise median) matrix of all patterns
        detected this way.
    """

    pattern_windows = []  # list containing all pannel of detected patterns
    pattern_sums = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # sum of all detected patterns
    agglomerated_pattern = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # median of all detected patterns
    detected_patterns = []
    n_patterns = 0

    for matrix, name, indices in zip(
        contact_map.sub_mats,
        contact_map.sub_mats_labels,
        contact_map.sub_mats_detectable_bins,
    ):
        plt.imshow(matrix.todense(), vmax = np.percentile(matrix.tocoo().data, 99)); plt.show()
        nr = matrix.shape[0]
        nc = matrix.shape[1]
        res2 = utils.corrcoef2d(matrix, kernel)  # !!  Here the pattern match  !!
        res2 = res2.tocoo()
        plt.imshow(res2.todense(), cmap="Reds"); plt.show()
        res2.data[np.isnan(res2.data)] = 0.0
        res2.eliminate_zeros()
        n2r = res2.shape[0]
        n2c = res2.shape[1]
        res_rescaled = sparse.csr_matrix(np.shape(matrix))
        res_rescaled[
            np.ix_(range(int(area), n2r + int(area)), range(int(area), n2c + int(area)))
        ] = res2
        VECT_VALUES = np.reshape(res_rescaled, (1, nr * nc))
        VECT_VALUES = VECT_VALUES[0]
        thr = np.median(VECT_VALUES) + precision * np.std(VECT_VALUES)
        indices_max = np.where(res_rescaled > thr)
        indices_max = np.array(indices_max)
        res_rescaled = np.triu(res_rescaled)
        res_rescaled[(res_rescaled) < 0] = 0
        pattern_peak = utils.picker(res_rescaled, thr)
        if pattern_peak != "NA":
            if pattern_type == "loops":
                # Assume all loops are not found too far-off in the matrix
                if not interchrom:
                    # NOTE: "Too far off" will depend on bin size here.
                    mask = np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1])) < 5000
                    pattern_peak = pattern_peak[mask, :]

                mask = np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1])) > 2
                pattern_peak = pattern_peak[mask, :]
            elif pattern_type == "borders":
                # Borders are always on the diagonal
                mask = np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1])) == 0
                pattern_peak = pattern_peak[mask, :]
            for l in pattern_peak:
                if l[0] in indices[0] and l[1] in indices[1]:
                    p1 = int(l[0])
                    p2 = int(l[1])
                    if p1 > p2:
                        p22 = p2
                        p2 = p1
                        p1 = p22
                    if (
                        p1 - area >= 0
                        and p1 + area + 1 < nr
                        and p2 - area >= 0
                        and p2 + area + 1 < nc
                    ):
                        window = matrix[
                            np.ix_(
                                range(p1 - area, p1 + area + 1),
                                range(p2 - area, p2 + area + 1),
                            )
                        ]
                        if (
                            len(window[window == 1.0])
                            < ((area * 2 + 1) ** 2) * undetermined_percentage / 100.0
                        ):  # there should not be many indetermined bins
                            n_patterns += 1
                            score = res_rescaled[l[0], l[1]]
                            detected_patterns.append((name, l[0], l[1], score))
                            pattern_sums += window
                            pattern_windows.append(window)
                        else:
                            detected_patterns.append((name, "NA", "NA", "NA"))
        else:
            detected_patterns.append((name, "NA", "NA", "NA"))

    # Computation of stats on the whole set - Agglomerated procedure :
    for i in range(0, area * 2 + 1):
        for j in range(0, area * 2 + 1):
            list_temp = []
            for el in range(1, len(pattern_windows)):
                list_temp.append(pattern_windows[el][i, j])
            agglomerated_pattern[i, j] = np.median(list_temp)

    nb_patterns = len(pattern_windows)
    return detected_patterns, agglomerated_pattern, nb_patterns


border_detector = functools.partial(
    pattern_detector, pattern_type="borders", undetermined_percentage=20.0
)

loop_detector = functools.partial(
    pattern_detector, pattern_type="loops", undetermined_percentage=1.0
)

PATTERN_DISPATCHER = {"loops": loop_detector, "borders": border_detector}
chromo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET_KERNEL_PATH = pathlib.Path(os.path.join(chromo_dir, "data"))


def load_kernels(pattern):
    """Load pattern kernels

    Look for one or several kernel file (in CSV format).

    Parameters
    ----------
    pattern : str
        The pattern type. Must be one of 'borders', 'loops' or 'centromeres',
        but partial matching is allowed.

    Returns
    -------
    pattern_kernels : list
        A list of array_likes corresponding to the loaded patterns.
    """

    pattern_path = pathlib.Path(pattern)
    if pattern in PATTERN_DISPATCHER.keys():
        pattern_globbing = PRESET_KERNEL_PATH.glob("*{}*".format(pattern))
    elif pattern_path.is_dir():
        pattern_globbing = pattern_path.glob("*")
    else:
        # Not finished ?
        pattern_globbing = (pattern_path,)
    for kernel_file in pattern_globbing:
        yield np.loadtxt(kernel_file, dtype=np.float)


def explore_patterns(
    contact_map,
    pattern_type="loops",
    custom_kernels=None,
    precision=4,
    iterations="auto",
    window=4,
):
    """Explore patterns in a list of matrices

    Given a pattern type, attempt to detect that pattern in each matrix with
    confidence determined by the precision parameter. The detection is done
    in a multi-pass process:
    - First, pattern matching is done with the initial supplied kernels.
    - Then, an 'agglomerated' median pattern from all previously detected
    patterns is generated, and detection is done using this pattern for
    matching instead.
    - Repeat as needed or until convergence.

    Parameters
    ----------
    contact_map : ContactMap object
        Object containing the Hi-C contact map and all intra- + inter- chromosomal
        sub matrices as well as other attributes.
    pattern_type : file, str or pathlib.Path
        The type of pattern to detect. Must be one of 'borders', 'loops', or
        'centromeres', but partial matching is allowed. If it looks like a
        path, instead the file is loaded and used as a tempalte itself.
    precision : float, optional
        The confidence with which pattern attribution is performed. The lower,
        the more detected patterns, the more false positives. Default is 4.
    iterations : str or int, optional
        How many iterations after the first kernel-based pass to perform in
        order to detect more patterns. If set to 'auto', iterations are
        performed until no more patterns are detected from one pass to the
        next.
    window : int, optional
        The pattern window area. When a pattern is discovered in a previous
        pass, further detected patterns falling into that area are discarded.

    Returns
    -------
    all_patterns : dict
        A dictionary in the form 'chromosome': list_of_coordinates_and_scores,
        and it is assumed that each matrix corresponds to a different
        chromosome. The chromosome string is determined by the matrix filename.
    agglomerated_patterns : list
        A list of agglomerated patterns after each pass.
    """

    # Dispatch detectors: the border detector has specificities while the
    # loop detector is more generic, so we use the generic one by default if
    # a pattern specific detector isn't implemented.
    custom_pattern_detector = PATTERN_DISPATCHER.get(pattern_type, loop_detector)

    if custom_kernels is None:
        chosen_kernels = load_kernels(pattern_type)
    else:
        chosen_kernels = load_kernels(custom_kernels)

    # Init parameters for the while loop:
    #   - There's always at least one iteration (with the kernel)
    #   - Loop stops when the same number of patterns are detected after an
    #     iterations, or the max number of iterations has been specified and
    #     reached.
    #   - After the first pass, instead of the starting kernel the
    #     'agglomerated pattern' is used for pattern matching.

    all_patterns = set()
    hashed_neighborhoods = set()
    agglomerated_patterns = [list(chosen_kernels)]
    iteration_count = 0
    old_pattern_count, current_pattern_count = -1, 0
    list_current_pattern_count = []
    # Depending on matrix resolution, a pattern may be smeared over several
    # pixels. This trimming function ensures that there won't be many patterns
    # clustering around one location.

    def neigh_hash(coords, window):
        chromosome, pos1, pos2, _ = coords
        if pos1 == "NA" or pos2 == "NA":
            return "NA"
        else:
            return (chromosome, int(pos1) // window, int(pos2) // window)

    # Loop on the patterns to convolve the matrices
    # condition for ending the loop. Meh
    while old_pattern_count != current_pattern_count:

        if iteration_count >= MAX_ITERATIONS or (
            iterations != "auto" and iteration_count >= iterations
        ):
            break

        agglomerated_patterns.append([])
        old_pattern_count = current_pattern_count
        iteration_count += 1

        for kernel in agglomerated_patterns[-2]:
            (
                detected_coords,
                agglomerated_pattern,
                nb_patterns,
            ) = custom_pattern_detector(contact_map, kernel, precision=precision)
            for new_coords in detected_coords:
                if neigh_hash(new_coords, window=window) not in hashed_neighborhoods:
                    chromosome, pos1, pos2, score = new_coords
                    if pos1 != "NA":
                        pos1 = int(pos1)
                    if pos2 != "NA":
                        pos2 = int(pos2)
                    all_patterns.add((chromosome, pos1, pos2, score))
                    hashed_neighborhoods.add(neigh_hash(new_coords, window=window))
            agglomerated_patterns[-1].append(agglomerated_pattern)
        current_pattern_count = nb_patterns
        list_current_pattern_count.append(current_pattern_count)

    return all_patterns, agglomerated_patterns, list_current_pattern_count


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

    detectable = utils.get_detectable_bins(matrix)
    matscn = utils.scn_func(matrix, detectable)
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
    print(emplacement)
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


def write_results(patterns_to_plot, output):
    for pattern in patterns_to_plot:
        file_name = pattern + ".txt"
        file_path = output / file_name
        with file_path.open("w") as outf:
            for tup in sorted(
                [tup for tup in patterns_to_plot[pattern] if "NA" not in tup]
            ):
                outf.write(" ".join(map(str, tup)) + "\n")


def agglomerated_plot(agglomerated_pattern, name="agglomerated patterns", output=None):
    """
    Plot the ???
    """
    if output is None:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    plt.imshow(
        agglomerated_pattern, interpolation="none", vmin=0.0, vmax=2.0, cmap="seismic"
    )
    plt.title("Ag {}".format(name))
    plt.colorbar()
    emplacement = output / pathlib.Path(name + ".pdf")
    plt.savefig(emplacement, dpi=100, format="pdf")
    plt.close("all")


def main():
    arguments = docopt.docopt(__doc__, version=__version__)

    # Parse command line arguments
    map_path = arguments["<contact_map>"]
    input_format = arguments["--input-format"]
    kernels = arguments["--kernels"]
    loops = arguments["--loops"]
    borders = arguments["--borders"]
    interchrom = arguments["--inter"]
    precision = float(arguments["--precision"])
    iterations = arguments["--iterations"]
    output = arguments["<output>"]
    list_current_pattern_count = []
    # If output is not specified, use current directory
    if not output:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    output.mkdir(exist_ok=True)

    try:
        iterations = int(iterations)
    except ValueError:
        if iterations != "auto":
            raise ValueError('Error! Iterations must be an integer or "auto"')

    patterns_to_explore = []
    # Read which patterns should be analysed
    if loops:
        patterns_to_explore.append("loops")
    if borders:
        patterns_to_explore.append("borders")
    if kernels:
        kernel_list = [k for k in kernels.split(",") if k]
    else:
        kernel_list = None

    patterns_to_plot = dict()
    agglomerated_to_plot = dict()
    # Load input file with appropriate function
    format_loader = {"bg2": io.load_bedgraph2d, "cool": io.load_cool}
    # Load contact map and chromosome start bins coords
    loaded_map, chroms = format_loader[input_format](map_path)
    contact_map = ContactMap(loaded_map, chroms)

    for pattern in patterns_to_explore:
        (
            all_patterns,
            agglomerated_patterns,
            list_current_pattern_count,
        ) = explore_patterns(
            contact_map,
            pattern,
            iterations=iterations,
            precision=precision,
            custom_kernels=kernel_list,
        )
        if contact_map.interchrom is not None:
            # Get index of patterns in full genome matrix.
            all_patterns = (
                contact_map.get_full_mat_pattern(pattern) for pattern in all_patterns
            )
            # all_patterns = map(utils.get_inter_idx, all_patterns)
        patterns_to_plot[pattern] = list(all_patterns)
        agglomerated_to_plot[pattern] = agglomerated_patterns

    write_results(patterns_to_plot, output)
    base_names = pathlib.Path(map_path).name

    # Iterate over each intra or inter sub-matrix
    for k, matrix in enumerate(contact_map.sub_mats):
        if isinstance(matrix, np.ndarray):
            pattern_plot(patterns_to_plot, matrix, output=output, name=k)
    # agglomerated_to_plot = {pattern('loop' or 'border'): [[]], ...}
    for (pattern, agglomerated_iter_list) in agglomerated_to_plot.items():
        # agglomerated_iter_list = [[np.array() (kernel at iteration i)]]
        for i, agglomerated_iteration in enumerate(agglomerated_iter_list):
            # agglomerated_iteration = [np.array() (kernel at iteration i)]
            for j, agglomerated_matrix in enumerate(agglomerated_iteration):
                # agglomerated_matrix = np.array()
                my_name = (
                    "Agglomerated {} of {} patterns iteration {} " "kernel {}"
                ).format(pattern, list_current_pattern_count[i - 1], i, j)
                agglomerated_plot(agglomerated_matrix, name=my_name, output=output)
    write_results(patterns_to_plot, output)

    return 0


if __name__ == "__main__":
    main()
