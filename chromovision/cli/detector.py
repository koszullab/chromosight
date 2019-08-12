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
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pathlib
import os
import functools
import docopt
from chromovision.version import __version__
from chromovision.utils.contacts_map import ContactMap
from scipy import sparse


def write_results(patterns_to_plot, output):
    for pattern in patterns_to_plot:
        file_name = pattern + ".txt"
        file_path = output / file_name
        with file_path.open("w") as outf:
            for tup in sorted(
                [tup for tup in patterns_to_plot[pattern] if "NA" not in tup]
            ):
                outf.write(" ".join(map(str, tup)) + "\n")


def main():
    arguments = docopt.docopt(__doc__, version=__version__)

    # Parse command line arguments
    map_path = arguments["<contact_map>"]
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

    patterns_types = []
    # Read which patterns should be analysed
    if loops:
        patterns_types.append("loops")
    if borders:
        patterns_types.append("borders")
    if kernels:
        kernel_list = [k for k in kernels.split(",") if k]
    else:
        kernel_list = None

    patterns_to_plot = dict()
    agglomerated_to_plot = dict()

    contact_map = ContactMap(map_path, interchrom)

    # Loop over types of patterns (loops, TADs, ...)
    for pattern_type in patterns_types:
        (
            all_patterns,
            agglomerated_patterns,
            list_current_pattern_count,
        ) = explore_patterns(
            contact_map,
            pattern_type,
            iterations=iterations,
            precision=precision,
            custom_kernels=kernel_list,
        )
        if contact_map.interchrom is not None:
            # Get bin indices of patterns in full genome matrix.
            all_patterns = (
                contact_map.get_full_mat_pattern(pattern) for pattern in all_patterns
            )
            # all_patterns = map(utils.get_inter_idx, all_patterns)
        patterns_to_plot[pattern_type] = list(all_patterns)
        agglomerated_to_plot[pattern_type] = agglomerated_patterns

    write_results(patterns_to_plot, output)
    # base_names = pathlib.Path(map_path).name

    # Iterate over each intra or inter sub-matrix
    for k, matrix in enumerate(contact_map.sub_mats):
        if isinstance(matrix, np.ndarray):
            pattern_plot(
                patterns_to_plot,
                matrix,
                output=output,
                name=contact_map.sub_mat_labels[k],
            )
    # agglomerated_to_plot = {pattern('loop' or 'border'): {iteration: [kernel, ...], ...}}
    for pattern_type, agglomerated_kernels in agglomerated_to_plot.items():
        # agglomerated_kernels_iter = [kernel1 at iteration i, kernel2 at iteration i, ...]
        for iteration, agglomerated_kernels_iter in agglomerated_kernels.items():
            # agglomerated_iteration = [np.array() (kernel at iteration i)]
            for kernel_id, agglomerated_matrix in enumerate(agglomerated_kernels_iter):
                # agglomerated_matrix = np.array()
                my_name = ("pileup_{}_{}_patterns_iteration_{}_kernel_{}").format(
                    pattern_type,
                    list_current_pattern_count[iteration - 1],
                    iteration,
                    kernel_id,
                )
                agglomerated_plot(agglomerated_matrix, name=my_name, output=output)
    write_results(patterns_to_plot, output)

    return 0


if __name__ == "__main__":
    main()
