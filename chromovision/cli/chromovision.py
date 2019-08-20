#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromovision detect <contact_map> [<output>] [--kernel-config FILE]
                        [--pattern=loops] [--precision=auto] [--iterations=auto]
                        [--inter] [--max-dist=auto]
    chromovision generate-config <prefix> [--preset loops]

    detect: 
        performs pattern detection on a Hi-C contact map using kernel convolution
    generate-config:
        Generate pre-filled config files to use for `chromovision detect`. 
        A config consists of a JSON file describing analysis parameters for the
        detection and path pointing to kernel matrices files. Those matrices
        files are tsv files with numeric values ordered in a square dense matrix
        to use for convolution.

Arguments for detect:
    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    contact_map                 The Hi-C contact map to detect patterns on, in
                                bedgraph2d or cool format. 
    output                      name of the output directory
    -I, --inter                 Enable to consider interchromosomal contacts.
    -i, --iterations auto       How many iterations to perform after the first
                                template-based pass. Auto sets an appropriate
                                value loaded from the kernel configuration
                                file. [default: auto]
    -k, --kernel-config FILE    Optionally give a path to a custom JSON kernel
                                config path. Use this to override pattern if 
                                you do not want to use one of the preset 
                                patterns.
    -m, --max-dist auto         Maximum distance from the diagonal (in base pairs)
                                at which pattern detection should operate. Auto
                                sets a value based on the kernel configuration
                                file and the signal to noise ratio. [default: auto]
    -P, --pattern loops         Which pattern to detect. This will use preset
                                configurations for the given pattern. Possible
                                values are: loops, borders, hairpin. [default: loops]
    -p, --precision auto        Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: auto]

Arguments for generate-config:
    prefix                      Path prefix for config files. If prefix is a/b,
                                files a/b.json and a/b.1.txt will be generated.
                                If a given pattern has N kernel matrices, N txt
                                files are created they will be named a/b.[1-N].txt.
    -p, --preset loops         Generate a preset config for the given pattern.
                                Preset configs available are "loops" and 
                                "borders". [default: loops]
"""
import numpy as np
import pathlib
import sys
import json
import docopt
from profilehooks import profile
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
            "{param_name} set to {default_val} based on config file.\n".format(
                default_val=config[param_name], param_name=param_name
            )
        )
    else:
        try:
            config[param_name] = param_type(param_value)
        except ValueError:
            raise ValueError(f'Error: {param_name} must be a {param_type} or "auto"')

    return config


def cmd_generate_config(arguments):
    # Parse command line arguments for generate_config
    prefix = arguments["<prefix>"]
    pattern = arguments["--preset"]
    arguments = docopt.docopt(__doc__, version=__version__)

    kernel_config = load_kernel_config(pattern, False)

    # Write kernel matrices to files with input prefix and replace kernels
    # by their path in config
    for mat_id, mat in enumerate(kernel_config["kernels"]):
        mat_path = f"{prefix}.{mat_id+1}.txt"
        np.savetxt(mat_path, mat)
        kernel_config["kernels"][mat_id] = mat_path

    # Write config to JSON file using prefix
    with open(f"{prefix}.json", "w") as config_handle:
        json.dump(kernel_config, config_handle, indent=4)


@profile
def cmd_detect(arguments):
    # Parse command line arguments for detect
    kernel_config_path = arguments["--kernel-config"]
    interchrom = arguments["--inter"]
    iterations = arguments["--iterations"]
    mat_path = arguments["<contact_map>"]
    max_dist = arguments["--max-dist"]
    pattern = arguments["--pattern"]
    precision = arguments["--precision"]

    output = arguments["<output>"]
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
    kernel_config = _override_kernel_config("max_dist", max_dist, int, kernel_config)

    # kernel_config = _override_kernel_config("max_dist", max_dist, int, kernel_config)
    # Make shorten max distance in case matrix is noisy

    patterns_to_plot = dict()
    contact_map = ContactMap(mat_path, interchrom, kernel_config["max_dist"])

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

    write_results(patterns_to_plot, kernel_config["name"], output)
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
    write_results(patterns_to_plot, kernel_config["name"], output)


def main():
    arguments = docopt.docopt(__doc__, version=__version__)
    detect = arguments["detect"]
    generate_config = arguments["generate-config"]
    if detect:
        cmd_detect(arguments)
    elif generate_config:
        cmd_generate_config(arguments)
    return 0


if __name__ == "__main__":
    main()
