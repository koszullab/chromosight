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
import pathlib
import docopt
from chromovision.version import __version__
from chromovision.utils.contacts_map import ContactMap
from chromovision.utils.io import write_results
from chromovision.utils.plotting import pattern_plot, pileup_plot
from chromovision.utils.detection import explore_patterns


def main():
    arguments = docopt.docopt(__doc__, version=__version__)

    # Parse command line arguments
    mat_path = arguments["<contact_map>"]
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
    pileup_to_plot = dict()
    contact_map = ContactMap(mat_path, interchrom)

    # Loop over types of patterns (loops, TADs, ...)
    for pattern_type in patterns_types:
        (all_patterns, pileup_patterns, list_current_pattern_count) = explore_patterns(
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
        pileup_to_plot[pattern_type] = pileup_patterns

    write_results(patterns_to_plot, output)
    # base_names = pathlib.Path(map_path).name

    # Iterate over each intra or inter sub-matrix
    for k, matrix in enumerate(contact_map.sub_mats):
        if isinstance(matrix, np.ndarray):
            pattern_plot(
                patterns_to_plot,
                matrix,
                output=output,
                name=contact_map.sub_mats_labels[k],
            )
    # pileup_to_plot = {pattern('loop' or 'border'): {iteration: [kernel, ...], ...}}
    for pattern_type, pileup_kernels in pileup_to_plot.items():
        # pileup_kernels_iter = [kernel1 at iteration i, kernel2 at iteration i, ...]
        for iteration, pileup_kernels_iter in pileup_kernels.items():
            # pileup_iteration = [np.array() (kernel at iteration i)]
            for kernel_id, pileup_matrix in enumerate(pileup_kernels_iter):
                # pileup_matrix = np.array()
                my_name = ("pileup_{}_{}_patterns_iteration_{}_kernel_{}").format(
                    pattern_type,
                    list_current_pattern_count[iteration - 1],
                    iteration,
                    kernel_id,
                )
                pileup_plot(pileup_matrix, name=my_name, output=output)
    write_results(patterns_to_plot, output)

    return 0


if __name__ == "__main__":
    main()
