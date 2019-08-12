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
    -k, --kernel-config None    A custom kernel template to use, if not using
                                one of the presets. If not supplied, the
                                loops or borders option must be used.
                                [default: None]
    -P, --pattern loops         Which pattern to detect. This will use preset
                                configurations for the given pattern. Possible
                                values are: loops, borders. [default: loops]
    -p, --precision auto        Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: auto]
    -I, --inter                 Use to consider interchromosomal contacts.
    -i, --iterations auto       How many iterations to perform after the first
                                template-based pass. Auto means an appropriate
                                value will be loaded from the kernel
                                configuration file. [default: auto]
"""
import numpy as np
import pathlib
import sys
import docopt
from chromovision.version import __version__
from chromovision.utils.contacts_map import ContactMap
from chromovision.utils.io import write_results, load_kernel_config
from chromovision.utils.plotting import pattern_plot, pileup_plot
from chromovision.utils.detection import explore_patterns


def _override_kernel_config(param_name, param_value, param_type, config):
    """
    Helper function to determine if config file value should be overriden by
    user.
    """

    if param_value == "auto":
        sys.stderr.write(
            "{param_name} set to {default_val} from config file.\n".format(
                default_val=config[param_name], param_name=param_name
            )
        )
    else:
        try:
            config[param_name] = param_type(param_value)
        except ValueError:
            raise ValueError(f'Error: {param_name} must be a {param_type} or "auto"')

    return config


def main():
    arguments = docopt.docopt(__doc__, version=__version__)

    # Parse command line arguments
    mat_path = arguments["<contact_map>"]
    kernel_config_path = arguments["--kernel-config"]
    pattern = arguments["--pattern"]
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

    # Read a user-provided kernel config if custom is true
    # Else, load a preset kernel config for input pattern
    # Configs are JSON files containing all parameter associated with the pattern
    # They are loaded into a dictionary in the form :
    # {"max_iterations": 3, "kernels": [kernel1, kernel2, ...], ...}
    # Where each kernel is a 2D numpy array representing the pattern
    if kernel_config_path is not None:
        custom = True
        # Loading input path as config
        config_path = kernel_config_path
    else:
        custom = False
        # Will use a preset config file matching pattern name
        config_path = pattern
    kernel_config = load_kernel_config(config_path, custom)

    # User can override configuration for input pattern if desired
    kernel_config = _override_kernel_config(
        "max_iterations", iterations, int, kernel_config
    )
    kernel_config = _override_kernel_config(
        "precision", precision, float, kernel_config
    )

    patterns_to_plot = dict()
    contact_map = ContactMap(mat_path, interchrom)

    # Loop over types of patterns (loops, TADs, ...)
    (all_patterns, pileup_patterns, list_current_pattern_count) = explore_patterns(
        contact_map, kernel_config
    )
    if contact_map.interchrom is not None:
        # Get bin indices of patterns in full genome matrix.
        all_patterns = (
            contact_map.get_full_mat_pattern(pattern) for pattern in all_patterns
        )
        # all_patterns = map(utils.get_inter_idx, all_patterns)
    patterns_to_plot = list(all_patterns)

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

    for iteration, pileup_kernels_iter in pileup_patterns.items():
        # pileup_iteration = [np.array() (kernel at iteration i)]
        for kernel_id, pileup_matrix in enumerate(pileup_kernels_iter):
            # pileup_matrix = np.array()
            my_name = ("pileup_{}_{}_patterns_iteration_{}_kernel_{}").format(
                kernel_config["name"],
                list_current_pattern_count[iteration - 1],
                iteration,
                kernel_id,
            )
            pileup_plot(pileup_matrix, name=my_name, output=output)
    write_results(patterns_to_plot, output)

    return 0


if __name__ == "__main__":
    main()
