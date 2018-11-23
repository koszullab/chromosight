#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    declooptor.py detect <contact_maps> [<output>] [--kernel=None] [--loops]
                         [--borders] [--precision=4] [--iterations=auto]
                         [--output]

Arguments:
    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    contact_maps                The Hi-C contact maps to detect patterns on, in
                                CSV format. File names must be separated by a
                                colon.
    -k None, kernel None        A custom kernel template to use, if not using
                                one of the presets. If not supplied, the
                                loops or borders option must be used.
                                [default: None]
    -L, --loops                 Whether to detect chromatin loops.
    -B, --borders               Whether to detect domain borders.
    -p 4, --precision 4         Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: 4]
    -i auto, --iterations auto  How many iterations to perform after the first
                                template-based pass. Auto means iterations are
                                performed until convergence. [default: auto]
    -o, --output                Output directory to write the detected pattern
                                coordinates, agglomerated plots and matrix
                                images into.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import utils
import pathlib
import itertools
import functools
import docopt

from version import __version__

MAX_ITERATIONS = 50


def pattern_detector(
    matrices,
    kernel,
    pattern_type="loops",
    precision=4.0,
    area=8,
    undetermined_percentage=1.,
    labels=None,
):

    if isinstance(matrices, np.ndarray):
        matrix_list = [matrices]
    else:
        matrix_list = matrices

    if labels is None:
        labels = range(len(matrix_list))
    elif isinstance(labels, str):
        labels = [labels]

    pattern_windows = []  # list containing all pannel of detected patterns
    pattern_sums = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # sum of all detected patterns
    agglomerated_pattern = np.zeros(
        (area * 2 + 1, area * 2 + 1)
    )  # median of all detected patterns
    detected_patterns = []
    n_patterns = 0

    for matrix, name in zip(matrix_list, labels):

        detrended, threshold_vector = utils.detrend(matrix)
        matrix_indices = np.where(matrix.sum(axis=0) > threshold_vector)
        n = matrix.shape[0]

        res2 = utils.corrcoef2d(
            detrended, kernel, centered_p=False
        )  # !!  Here the pattern match  !!
        res2[np.isnan(res2)] = 0.0
        n2 = res2.shape[0]
        res_rescaled = np.zeros(np.shape(matrix))
        res_rescaled[
            np.ix_(
                range(int(area), n2 + int(area)),
                range(int(area), n2 + int(area)),
            )
        ] = res2
        VECT_VALUES = np.reshape(res_rescaled, (1, n ** 2))
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
                mask = (
                    np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1]))
                    < 5000
                )
                pattern_peak = pattern_peak[mask, :]
                mask = (
                    np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1])) > 2
                )
                pattern_peak = pattern_peak[mask, :]
            elif pattern_type == "borders":
                # Borders are always on the diagonal
                mask = (
                    np.array(abs(pattern_peak[:, 0] - pattern_peak[:, 1])) == 0
                )
                pattern_peak = pattern_peak[mask, :]
            for l in pattern_peak:
                if l[0] in matrix_indices[0] and l[1] in matrix_indices[0]:
                    p1 = int(l[0])
                    p2 = int(l[1])
                    if p1 > p2:
                        p22 = p2
                        p2 = p1
                        p1 = p22
                    if (
                        p1 - area >= 0
                        and p1 + area + 1 < n
                        and p2 - area >= 0
                        and p2 + area + 1 < n
                    ):
                        window = detrended[
                            np.ix_(
                                range(p1 - area, p1 + area + 1),
                                range(p2 - area, p2 + area + 1),
                            )
                        ]
                        if (
                            len(window[window == 1.])
                            < ((area * 2 + 1) ** 2)
                            * undetermined_percentage
                            / 100.
                        ):  # there should not be many indetermined bins
                            n_patterns += 1
                            score = res_rescaled[l[0], l[1]]
                            detected_patterns.append([name, l[0], l[1], score])
                            pattern_sums += window
                            pattern_windows.append(window)
                        else:
                            detected_patterns.append([name, "NA", "NA", "NA"])
        else:
            detected_patterns.append([name, "NA", "NA", "NA"])

    # Computation of stats on the whole set - Agglomerated procedure :
    for i in range(0, area * 2 + 1):
        for j in range(0, area * 2 + 1):
            list_temp = []
            for el in range(1, len(pattern_windows)):
                list_temp.append(pattern_windows[el][i, j])
            agglomerated_pattern[i, j] = np.median(list_temp)

    return detected_patterns, agglomerated_pattern


border_detector = functools.partial(
    pattern_detector, pattern_type="borders", undetermined_percentage=20.
)
loop_detector = functools.partial(
    pattern_detector, pattern_type="loops", undetermined_percentage=1.
)

PATTERN_DISPATCHER = {"loops": loop_detector, "borders": border_detector}
PRESET_KERNEL_PATH = pathlib.Path("/home/rsg/DATA/Loops_Detector/data")


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
    if pattern_path.is_dir():
        pattern_globbing = pattern_path.glob("*")
    elif pattern in PATTERN_DISPATCHER.keys():
        print("cas spécial")
        pattern_globbing = PRESET_KERNEL_PATH.glob("*{}*".format(pattern))
    else:
        pattern_globbing = (pattern_path,)
    pattern_kernels = [
        np.loadtxt(kernel_file) for kernel_file in pattern_globbing
    ]
    print(pattern_globbing)
    print(pattern_kernels)
    return pattern_kernels


def explore_patterns(
    matrices,
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
    matrices : iterable
        A list (or similarly iterable object) of matrices to detect patterns
        on.
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
    my_pattern_detector = PATTERN_DISPATCHER.get(pattern_type, loop_detector)

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
    agglomerated_patterns = []
    agglomerated_pattern = None
    iteration_count = 0
    old_pattern_count, current_pattern_count = -1, 0
    # Depending on matrix resolution, a pattern may be smeared over several
    # pixels. This trimming function ensures that there won't be many patterns
    # clustering around one location.

    def clean_by_neighborhood(coord_list):
        for my_pattern in coord_list:

            chromosome, pos1, pos2, _ = my_pattern
            if pos1 != "NA":
                pos1 = int(pos1)
                pos2 = int(pos2)

                neighbours = set(
                    zip(
                        itertools.repeat(chromosome),
                        itertools.product(
                            range(pos1 - window, pos1 + window + 1),
                            range(pos2 - window, pos2 + window + 1),
                        ),
                    )
                )

                if not neighbours.intersection(all_patterns):
                    yield (chromosome, pos1, pos2)

    print(len(chosen_kernels))
    while old_pattern_count != current_pattern_count:
        print(old_pattern_count, current_pattern_count)

        if iteration_count >= MAX_ITERATIONS or (
            iterations != "auto" and iteration_count >= iterations
        ):
            print("J'ai fini")
            break

        old_pattern_count = current_pattern_count
        print(all_patterns)
        for kernel in chosen_kernels:
            if agglomerated_pattern is None:
                (detected_coords, agglomerated_pattern) = my_pattern_detector(
                    matrices, kernel, precision=precision
                )
                for new_coords in clean_by_neighborhood(detected_coords):
                    all_patterns.add(new_coords)
                    try:
                        agglomerated_patterns[-1].append(agglomerated_pattern)
                    except IndexError:
                        agglomerated_patterns.append([agglomerated_pattern])
            else:
                iteration_count += 1
                print("J'itère sur l'agglomerated plot")
                (detected_coords, agglomerated_pattern) = my_pattern_detector(
                    matrices, agglomerated_pattern, precision=precision
                )
            for new_coords in clean_by_neighborhood(detected_coords):
                all_patterns.add(new_coords)
                try:
                    agglomerated_patterns[-1].append(agglomerated_pattern)
                except IndexError:
                    agglomerated_patterns.append([agglomerated_pattern])

        current_pattern_count = len(all_patterns)

    return all_patterns, agglomerated_patterns


def pattern_plot(patterns, matrix, name=None, output=None):

    if name is None:
        name = "matrix"
    if output is None:
        output = pathlib.Path()

    th_sum = np.median(matrix.sum(axis=0)) - 2.0 * np.std(matrix.sum(axis=0))
    matscn = utils.scn_func(matrix, th_sum)
    plt.imshow(matscn ** 0.15, interpolation="none", cmap="afmhot_r")
    plt.title(name, fontsize=8)
    plt.colorbar()

    for pattern_type, all_patterns in patterns.items():
        if pattern_type == "borders":
            for border in all_patterns:
                if border[0] != name:
                    continue
                if border[1] != "NA":
                    pos1, pos2 = border
                    plt.plot(pos1, pos2, "D", color="white", markersize=0.5)
        elif pattern_type == "loops":
            for loop in all_patterns:
                if loop[0] != name:
                    continue
                if loop[1] != "NA":
                    pos1, pos2 = loop
                    plt.scatter(
                        pos1, pos2, s=15, facecolors="none", edgecolors="gold"
                    )

    plt.savefig(output / pathlib.Path(name) + ".pdf2", dpi=100, format="pdf")
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

    for matrix, name in zip(matrices, labels):
        dist = utils.distance_law(matrix)
        x = np.arange(0, len(dist))
        y = dist
        y[np.isnan(y)] = 0.
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


def agglomerated_plot(agglomerated_pattern, pattern_type="patterns"):

    plt.imshow(
        agglomerated_pattern,
        interpolation="none",
        vmin=0.,
        vmax=2.,
        cmap="seismic",
    )
    plt.colorbar()
    plt.title("Agglomerated plot of detected {}".format(pattern_type))
    plt.savefig(
        "agglomerated_{}.pdf".format(pattern_type), dpi=100, format="pdf"
    )
    plt.close("all")


def main():
    arguments = docopt.docopt(__doc__, version=__version__)

    contact_maps = arguments["<contact_maps>"].split(",")
    kernel = arguments["--kernel"]
    loops = arguments["--loops"]
    borders = arguments["--borders"]
    precision = float(arguments["--precision"])
    iterations = arguments["--iterations"]
    output = arguments["<output>"]

    if not output:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)

    output.mkdir(exist_ok=True)

    try:
        iterations = int(iterations)
    except ValueError:
        if iterations != "auto":
            print('Error! Iterations must be an integer or "auto"')
            return

    patterns_to_explore = set()
    if loops:
        patterns_to_explore.add("loops")
    if borders:
        patterns_to_explore.add("borders")

    patterns_to_plot = dict()

    loaded_maps = [np.loadtxt(contact_map) for contact_map in contact_maps]
    for pattern in patterns_to_explore:
        all_patterns, agglomerated_patterns = explore_patterns(
            loaded_maps, pattern, precision=precision
        )
        patterns_to_plot[pattern] = all_patterns
    for matrix in loaded_maps:
        pattern_plot(patterns_to_plot, matrix, name="test", output=output)


if __name__ == "__main__":
    main()
