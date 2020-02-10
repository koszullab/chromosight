#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops] [--precision=auto]
                        [--iterations=auto] [--resize-kernel] [--win-fmt={json,npy}]
                        [--subsample=no] [--inter] [--smooth-trend] [--n-mads=5]
                        [--min-dist=0] [--max-dist=auto] [--no-plotting]
                        [--min-separation=auto] [--threads=1] [--dump=DIR]
                        [--perc-undetected=auto] <contact_map> [<output>]
    chromosight generate-config <prefix> [--preset loops] [--click contact_map] [--win-size=auto] [--n-mads=INT]
    chromosight quantify [--inter] [--pattern=loops] [--subsample=no] [--win-fmt=json]
                         [--n-mads=5] [--win-size=auto] <bed2d> <contact_map> <output>
    chromosight test

    detect: 
        performs pattern detection on a Hi-C contact map using kernel convolution
    generate-config:
        Generate pre-filled config files to use for `chromosight detect`. 
        A config consists of a JSON file describing analysis parameters for the
        detection and path pointing to kernel matrices files. Those matrices
        files are tsv files with numeric values ordered in a square dense matrix
        to use for convolution.
    quantify:
        Given a list of pairs of positions and a contact map, computes the
        correlation coefficients between those positions and the kernel of the
        selected pattern.
    test:                       
        Download example data and run the program on it.

Arguments for detect:
    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    contact_map                 The Hi-C contact map to detect patterns on, in
                                bedgraph2d or cool format. 
    output                      name of the output directory
    -d, --dump=DIR              Directory where to save matrix dumps during
                                processing and detection. Each dump is saved as
                                a compressed npz of a sparse matrix and can be
                                loaded using scipy.sparse.load_npz. Disabled
                                by default.
    -I, --inter                 Enable to consider interchromosomal contacts.
                                Warning: Experimental feature with very high
                                memory consumption is very high, only use with
                                small matrices.
    -i, --iterations=auto       How many iterations to perform after the first
                                template-based pass. Auto sets an appropriate
                                value loaded from the kernel configuration
                                file. [default: 1]
    -k, --kernel-config=FILE    Optionally give a path to a custom JSON kernel
                                config path. Use this to override pattern if 
                                you do not want to use one of the preset 
                                patterns.
    -m, --min-dist=auto         Minimum distance from the diagonal (in base pairs).
                                If this value is smaller than the kernel size, the
                                kernel will be cropped to avoid overlapping the
                                diagonal, up to a min.size of 7x7. [default: auto]
    -M, --max-dist=auto         Maximum distance from the diagonal (in base pairs)
                                at which pattern detection should operate. Auto
                                sets a value based on the kernel configuration
                                file and the signal to noise ratio. [default: auto]
    -n, --no-plotting           Disable generation of pileup plots.
    -N, --n-mads=5              Maximum number of median absolute deviations below
                                the median of the logged bin sums distribution
                                allowed to consider detectable bins. [default: 5]
    -P, --pattern=loops         Which pattern to detect. This will use preset
                                configurations for the given pattern. Possible
                                values are: loops, borders, hairpins and
                                centromeres. [default: loops]
    -p, --precision=auto        Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: auto]
    -r, --resize-kernel         Experimental: Enable to resize kernel based on
                                input resolution.
    -s, --subsample=INT         If greater than 1, subsample contacts from the 
                                matrix to INT contacts. If between 0 and 1, subsample
                                a proportion of contacts instead. This is useful
                                when comparing matrices with different
                                coverages. [default: no]
    -S, --min-separation=auto   Minimum distance required between patterns, in
                                basepairs. If two patterns are closer than this
                                distance in both axes, the one with the lowest
                                score is discarded. [default: auto]
    -t, --threads=1             Number of CPUs to use in parallel. [default: 1]
    -T, --smooth-trend          Use isotonic regression to reduce noise at long
                                ranges caused by detrending. Do not enable this
                                for circular genomes.
    -u, --perc-undetected=auto  Maximum percentage of empty pixels in windows
                                allowed to keep detected patterns. [default: auto]
    -w, --win-fmt={json,npy}    File format used to store individual windows
                                around each pattern. Window order match
                                patterns inside the associated text file.
                                Possible formats are json and npy. [default: json]

Arguments for generate-config:
    prefix                      Path prefix for config files. If prefix is a/b,
                                files a/b.json and a/b.1.txt will be generated.
                                If a given pattern has N kernel matrices, N txt
                                files are created they will be named a/b.[1-N].txt.
    -e, --preset=loops          Generate a preset config for the given pattern.
                                Preset configs available are "loops" and 
                                "borders". [default: loops]
    -c, --click contact_map     Show input contact map and uses double clicks from
                                user to build the kernel. Warning: memory-heavy,
                                reserve for small genomes or subsetted matrices.
                                
Arguments for quantify:
    bed2d                       Tab-separated text files with columns chrom1, start1
                                end1, chrom2, start2, end2. Each line correspond to
                                a pair of positions (i.e. a position in the matrix).
    contact_map                 Path to the contact map, in bedgraph2d or
                                cool format.
    output                      output directory where files should be generated.
    -W, --win-size=auto         Window size, in basepairs, in which to compute the
                                correlation. The pattern kernel will be resized to
                                match this size. If the pattern must be enlarged,
                                linear interpolation is used to fill between pixels.
                                If not specified, the default kernel size will
                                be used instead. [default: auto]

"""
import numpy as np
import pandas as pd
import pathlib
import os
import io
from contextlib import contextmanager
import sys
import json
import docopt
import multiprocessing as mp
from chromosight.version import __version__
from chromosight.utils.contacts_map import HicGenome
import chromosight.utils.io as cio
import chromosight.utils.detection as cid
from chromosight.utils.plotting import pileup_plot, click_finder
from chromosight.utils.preprocessing import resize_kernel, crop_kernel
import scipy.stats as ss
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

URL_EXAMPLE_DATASET = (
    "https://raw.githubusercontent.com/koszullab/"
    "chromosight/master/data_test/example.bg2"
)

TEST_LOG = f"""Fetching test dataset at {URL_EXAMPLE_DATASET}...
Running detection on test dataset...
precision set to 3 based on config file.
max_dist set to 500000 based on config file.
min_dist set to 5000 based on config file.
min_separation set to 5000 based on config file.
max_perc_undetected set to 10.0 based on config file.
Found 690 / 720 detectable bins
Whole genome matrix normalized
Preprocessing sub-matrices...
Sub matrices extracted
Detecting patterns...
Minimum pattern separation is : 5
56 patterns detected
"""


def _override_kernel_config(param_name, param_value, param_type, config):
    """
    Helper function to determine if a config file value should be overriden
    by the user.
    """

    if param_value == "auto":
        try:
            sys.stderr.write(
                f"{param_name} set to {config[param_name]} based on config file.\n"
            )
        except KeyError:
            raise KeyError(
                f"{param_name} is not defined in the config. Please add it to "
                f"the JSON config file, or provide it as a command line option."
            )
    else:
        try:
            config[param_name] = param_type(param_value)
        except ValueError:
            raise ValueError(
                f'Error: {param_name} must be a {param_type} or "auto"'
            )

    return config


def cmd_quantify(arguments):
    bed2d_path = arguments["<bed2d>"]
    mat_path = arguments["<contact_map>"]
    output = pathlib.Path(arguments["<output>"])
    n_mads = float(arguments["--n-mads"])
    pattern = arguments["--pattern"]
    inter = arguments["--inter"]
    win_size = arguments["--win-size"]
    if win_size != "auto":
        win_size = int(win_size)
    subsample = arguments["--subsample"]
    # Create directory if it does not exist
    if not output.exists():
        os.makedirs(output, exist_ok=True)
    # Load 6 cols from 2D BED file and infer header
    bed2d = cio.load_bed2d(bed2d_path)
    # Warn user if --inter is disabled but list contains inter patterns
    if not inter and len(bed2d.start1[bed2d.chrom1 != bed2d.chrom2]) > 0:
        sys.stderr.write(
            "Warning: The bed2d file contains interchromosomal patterns. "
            "These patterns will not be scanned unless --inter is used.\n"
        )
    # Parse kernel config
    cfg = cio.load_kernel_config(pattern, False)
    # Instantiate and preprocess contact map
    hic_genome = HicGenome(mat_path, inter=inter, kernel_config=cfg)
    # enforce full scanning distance in kernel config
    cfg["max_dist"] = (
        hic_genome.matrix.shape[0] * hic_genome.resolution
    )
    cfg["min_dist"] = 0
    # Notify contact map instance of changes in scanning distance
    hic_genome.kernel_config = cfg
    # Subsample Hi-C contacts from the matrix, if requested
    if subsample != "no":
        hic_genome.subsample(subsample)
    # Normalize (balance) matrix using ICE
    hic_genome.normalize(n_mads)
    # Define how many diagonals should be used in intra-matrices
    hic_genome.compute_max_dist()
    # Split whole genome matrix into intra- and inter- sub matrices. Each sub
    # matrix is processed on the fly (obs / exp, trimming diagonals > max dist)
    hic_genome.make_sub_matrices()
    # Initialize output structures
    bed2d["score"] = 0.0
    positions = bed2d.copy()
    if win_size != "auto":
        km = kn = win_size
    else:
        km, kn = cfg["kernels"][0].shape
    windows = np.zeros((positions.shape[0], km, kn))
    # For each position, we use the center of the BED interval
    positions["pos1"] = (positions.start1 + positions.end1) // 2
    positions["pos2"] = (positions.start2 + positions.end2) // 2
    # Use each kernel matrix available for the pattern
    for kernel_id, kernel_matrix in enumerate(cfg["kernels"]):
        # Only resize kernel matrix if explicitely requested
        if win_size != "auto":
            kernel_matrix = resize_kernel(kernel_matrix, factor=win_size / km)
        kh = (km - 1) // 2
        kw = (kn - 1) // 2
        # Iterate over intra- and inter-chromosomal sub-matrices
        for sub_mat in hic_genome.sub_mats.iterrows():
            mat = sub_mat[1]
            # Filter patterns falling onto this sub-matrix
            sub_pat = positions.loc[
                (positions.chrom1 == mat.chr1) & (positions.chrom2 == mat.chr2)
            ]
            sub_pat_idx = sub_pat.index.values
            # Convert genomic coordinates to bins for horizontal and vertical axes
            for ax in [1, 2]:
                sub_pat_ax = sub_pat.loc[:, [f"chrom{ax}", f"pos{ax}"]].rename(
                    columns={f"chrom{ax}": "chrom", f"pos{ax}": "pos"}
                )
                sub_pat_bins = hic_genome.coords_to_bins(sub_pat_ax)
                sub_pat[f"bin{ax}"] = sub_pat_bins

            # Check for nan bins (coords that do not match any Hi-C fragments
            fall_out = np.isnan(sub_pat['bin1']) | np.isnan(sub_pat['bin2'])
            if np.any(fall_out):
                n_out = len(sub_pat_bins[fall_out])
                sys.stderr.write(
                    f"{n_out} entr{'ies' if n_out > 1 else 'y'} outside "
                    "genomic coordinates of the Hi-C matrix will be ignored.\n"
                )
            # Convert bins from whole genome matrix to sub matrix
            sub_pat = hic_genome.get_sub_mat_pattern(
                mat.chr1, mat.chr2, sub_pat
            )
            m = mat.contact_map.matrix.tocsr()
            # Iterate over patterns from the 2D BED file
            for i, x, y in zip(sub_pat_idx, sub_pat.bin1, sub_pat.bin2):
                # Check if the window goes out of bound
                if  np.all(np.isfinite([x, y])) and (
                    x - kh >= 0
                    and x + kh + 1 < m.shape[0]
                    and y - kw >= 0
                    and y + kw + 1 < m.shape[1]
                ):
                    x = int(x)
                    y = int(y)
                    # For each pattern, compute correlation score with all kernels
                    # but only keep the best
                    win = m[x - kh : x + kh + 1, y - kw : y + kw + 1].toarray()
                    try:
                        score = ss.pearsonr(
                            win.flatten(), kernel_matrix.flatten()
                        )[0]
                    # In case of NaNs introduced by division by 0 during detrend
                    except ValueError:
                        score = 0
                    if score > bed2d["score"][i] or kernel_id == 0:
                        bed2d["score"][i] = score
                # Pattern falls outside or at the edge of the matrix
                else:
                    win = np.zeros((km, kn))
                    bed2d["score"][i] = np.nan
                if kernel_id == 0:
                    windows[i, :, :] = win
        bed2d.to_csv(
            output / f"{pattern}_quant.txt", sep="\t", header=True, index=False
        )
        cio.save_windows(
            windows,
            f"{pattern}_quant",
            output_dir=output,
            format=arguments["--win-fmt"],
        )
        # with open(output / f"{pattern}_quant.json", "w") as win_handle:
        #    windows = {idx: win for idx, win in enumerate(windows)}
        #    json.dump(windows, win_handle, indent=4)


def cmd_generate_config(arguments):
    # Parse command line arguments for generate_config
    prefix = arguments["<prefix>"]
    pattern = arguments["--preset"]
    click_find = arguments["--click"]
    n_mads = float(arguments["--n-mads"])
    win_size = arguments["--win-size"]

    cfg = cio.load_kernel_config(pattern, False)

    # If prefix involves a directory, create it
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # If a specific window size if requested, resize all kernels 
    if win_size != "auto":
        win_size = int(win_size)
        resize = lambda m: resize_kernel(m, factor=win_size / m.shape[0])
        cfg['kernels'] = [resize(k) for k in cfg['kernels']]
    # Otherwise, just inherit window size from the kernel config
    else:
        win_size = cfg["kernels"][0].shape[0]

    # If click mode is enabled, build a kernel from scratch using
    # graphical display, otherwise, just inherit the pattern's kernel
    if click_find:
        hic_genome = HicGenome(
            click_find,
            inter=True,
            kernel_config=cfg,
        )
        # Normalize (balance) the whole genome matrix
        hic_genome.normalize(n_mads=n_mads)
        # enforce full scanning distance in kernel config
        
        hic_genome.max_dist = hic_genome.matrix.shape[0] * hic_genome.resolution
        # Process each sub-matrix individually (detrend diag for intra)
        hic_genome.make_sub_matrices()
        processed_mat = hic_genome.gather_sub_matrices().tocsr()
        windows = click_finder(processed_mat, half_w=int((win_size - 1) / 2))
        # Pileup all recorded windows and convert to JSON serializable list
        pileup = ndi.gaussian_filter(cid.pileup_patterns(windows), 1)
        cfg['kernels'] = [pileup.tolist()]
        # Show the newly generate kernel to the user, use zscore to highlight contrast
        hm = plt.imshow(
                np.log(pileup),
                vmax=np.percentile(pileup, 99),
                cmap='afmhot_r',
        )
        cbar = plt.colorbar(hm)
        cbar.set_label('Log10 Hi-C contacts')
        plt.title("Manually generated kernel")
        plt.show()
    # Write kernel matrices to files with input prefix and replace kernels
    # by their path in config
    for mat_id, mat in enumerate(cfg["kernels"]):
        mat_path = f"{prefix}.{mat_id+1}.txt"
        np.savetxt(mat_path, mat)
        cfg["kernels"][mat_id] = mat_path

    # Write config to JSON file using prefix
    with open(f"{prefix}.json", "w") as config_handle:
        json.dump(cfg, config_handle, indent=4)


def _detect_sub_mat(data):
    sub = data[0][1]
    config = data[1]
    kernel = data[2]
    dump = data[3]
    chrom_patterns, chrom_windows = cid.pattern_detector(
        sub.contact_map, config, kernel, dump
    )
    return {
        "coords": chrom_patterns,
        "windows": chrom_windows,
        "chr1": sub.chr1,
        "chr2": sub.chr2,
    }


def cmd_detect(arguments):
    # Parse command line arguments for detect
    kernel_config_path = arguments["--kernel-config"]
    dump = arguments["--dump"]
    interchrom = arguments["--inter"]
    iterations = arguments["--iterations"]
    mat_path = arguments["<contact_map>"]
    max_dist = arguments["--max-dist"]
    min_dist = arguments["--min-dist"]
    min_separation = arguments["--min-separation"]
    n_mads = float(arguments["--n-mads"])
    pattern = arguments["--pattern"]
    perc_undetected = arguments["--perc-undetected"]
    precision = arguments["--precision"]
    resize = arguments["--resize-kernel"]
    threads = arguments["--threads"]
    output = arguments["<output>"]
    win_fmt = arguments["--win-fmt"]
    subsample = arguments["--subsample"]
    if subsample == "no":
        subsample = None
    plotting_enabled = False if arguments["--no-plotting"] else True
    smooth_trend = arguments["--smooth-trend"]
    if smooth_trend is None:
        smooth_trend = False
    # If output is not specified, use current directory
    if not output:
        output = pathlib.Path()
    else:
        output = pathlib.Path(output)
    output.mkdir(exist_ok=True)

    if win_fmt not in ["npy", "json"]:
        sys.stderr.write("Error: --win-fmt must be either json or npy.\n")
        sys.exit(1)
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

    ### 0: LOAD INPUT
    params = {
        "max_iterations": (iterations, int),
        "precision": (precision, float),
        "max_dist": (max_dist, int),
        "min_dist": (min_dist, int),
        "min_separation": (min_separation, int),
        "max_perc_undetected": (perc_undetected, float),
    }
    cfg = cio.load_kernel_config(config_path, custom)
    for param_name, (param_value, param_type) in params.items():
        cfg = _override_kernel_config(
            param_name, param_value, param_type, cfg
        )

    if interchrom:
        sys.stderr.write(
            "WARNING: Detection on interchromosomal matrices is expensive in RAM\n"
        )
    hic_genome = HicGenome(
        mat_path,
        inter=interchrom,
        kernel_config=cfg,
        dump=dump,
        smooth=smooth_trend,
    )
    ### 1: Process input signal
    # Make necessary kernel adjustments
    for i, mat in enumerate(cfg["kernels"]):
        min_size, max_size = 7, 101
        new_kernel = mat
        # Adapt size of kernel matrices based on the signal resolution if requested
        if resize:
            new_kernel = resize_kernel(
                mat,
                kernel_res=cfg["resolution"],
                signal_res=hic_genome.resolution,
                min_size=min_size,
                max_size=max_size
            )
        # Crop the kernel if it is larger than min-dist and goes over diagonal
        # Do not trim if patterns of interest are on the diagonal (e.g. borders, hairpins)
        min_dist_diag = int(np.ceil(cfg['min_dist'] / hic_genome.resolution))
        # Make sure kernel is not too small or too large
        min_dist_diag = max(min_size, min_dist_diag)
        min_dist_diag = min(max_size, min_dist_diag)

        if min_dist_diag < max(mat.shape) and min_dist_diag > 0:
            new_kernel = crop_kernel(
                new_kernel,
                target_size=(min_dist_diag, min_dist_diag)
            )
            m, n = new_kernel.shape
            sys.stderr.write(
                'WARNING: --min-dist smaller than kernel size. Kernel has '
                f'been cropped to {m}x{n} to avoid overlapping the diagonal.\n'
            )
        cfg["kernels"][i] = new_kernel

    hic_genome.kernel_config = cfg
    # Subsample Hi-C contacts from the matrix, if requested
    # NOTE: Subsampling has to be done before normalisation
    hic_genome.subsample(subsample)
    # Normalize (balance) matrix using ICE
    hic_genome.normalize(n_mads=n_mads)
    # Define how many diagonals should be used in intra-matrices
    hic_genome.compute_max_dist()
    # Split whole genome matrix into intra- and inter- sub matrices. Each sub
    # matrix is processed on the fly (obs / exp, trimming diagonals > max dist)
    hic_genome.make_sub_matrices()

    all_pattern_coords = []
    all_pattern_windows = []

    ### 2: DETECTION ON EACH SUBMATRIX
    pool = mp.Pool(int(threads))
    n_sub_mats = hic_genome.sub_mats.shape[0]
    # Loop over the different kernel matrices for input pattern
    run_id = 0
    total_runs = (
        len(cfg["kernels"]) * cfg["max_iterations"]
    )
    sys.stderr.write("Detecting patterns...\n")
    for kernel_id, kernel_matrix in enumerate(cfg["kernels"]):
        # Adjust kernel iteratively
        for i in range(cfg["max_iterations"]):
            cio.progress(
                run_id, total_runs, f"Kernel: {kernel_id}, Iteration: {i}\n"
            )

            # Apply detection procedure to all sub matrices in parallel
            sub_mat_data = zip(
                hic_genome.sub_mats.iterrows(),
                [cfg for i in range(n_sub_mats)],
                [kernel_matrix for i in range(n_sub_mats)],
                [dump for i in range(n_sub_mats)],
            )
            # Run detection in parallel on different sub matrices, and show progress when
            # gathering results
            sub_mat_results = []
            for i, result in enumerate(pool.imap_unordered(_detect_sub_mat, sub_mat_data, 1)):
                chr1 = hic_genome.sub_mats.chr1[i]
                chr2 = hic_genome.sub_mats.chr2[i]
                cio.progress(i, n_sub_mats, f"{chr1}-{chr2}")
                sub_mat_results.append(result)
            #sub_mat_results = map(_detect_sub_mat, sub_mat_data)
            # Convert coordinates from chromosome to whole genome bins
            kernel_coords = [
                hic_genome.get_full_mat_pattern(
                    d["chr1"], d["chr2"], d["coords"]
                )
                for d in sub_mat_results
                if d["coords"] is not None
            ]

            # Gather newly detected pattern coordinates
            try:
                # Extract surrounding windows for each sub_matrix
                kernel_windows = np.concatenate(
                    [
                        w["windows"]
                        for w in sub_mat_results
                        if w["windows"] is not None
                    ],
                    axis=0,
                )
                all_pattern_coords.append(
                    pd.concat(kernel_coords, axis=0).reset_index(drop=True)
                )
                # Add info about kernel and iteration which detected these patterns
                all_pattern_coords[-1]["kernel_id"] = kernel_id
                all_pattern_coords[-1]["iteration"] = i
                all_pattern_windows.append(kernel_windows)

            # If no pattern was found with this kernel
            # skip directly to the next one, skipping iterations
            except ValueError:
                break

            # Update kernel with patterns detected at current iteration
            kernel_matrix = cid.pileup_patterns(kernel_windows)
            run_id += 1
    cio.progress(run_id, total_runs, f"Kernel: {kernel_id}, Iteration: {i}\n")

    # If no pattern detected on any chromosome, with any kernel, exit gracefully
    if len(all_pattern_coords) == 0:
        sys.stderr.write("No pattern detected ! Exiting.\n")
        sys.exit(0)

    # Combine patterns of all kernel matrices into a single array
    all_pattern_coords = pd.concat(all_pattern_coords, axis=0).reset_index(
        drop=True
    )
    # Combine all windows from different kernels into a single pile of windows
    all_pattern_windows = np.concatenate(all_pattern_windows, axis=0)

    # Compute minimum separation in bins and make sure it has a reasonable value
    separation_bins = int(
        cfg["min_separation"] // hic_genome.resolution
    )
    if separation_bins < 1:
        separation_bins = 1
    print(f"Minimum pattern separation is : {separation_bins}")
    # Remove patterns with overlapping windows (smeared patterns)
    distinct_patterns = cid.remove_neighbours(
        all_pattern_coords, win_size=separation_bins
    )

    # Drop patterns that are too close to each other
    all_pattern_coords = all_pattern_coords.loc[distinct_patterns, :]
    all_pattern_windows = all_pattern_windows[distinct_patterns, :, :]

    # Get from bins into basepair coordinates
    coords_1 = hic_genome.bins_to_coords(all_pattern_coords.bin1).reset_index(
        drop=True
    )
    coords_1.columns = [str(col) + "1" for col in coords_1.columns]
    coords_2 = hic_genome.bins_to_coords(all_pattern_coords.bin2).reset_index(
        drop=True
    )
    coords_2.columns = [str(col) + "2" for col in coords_2.columns]

    all_pattern_coords = pd.concat(
        [all_pattern_coords.reset_index(drop=True), coords_1, coords_2], axis=1
    )

    # Filter patterns closer than minimum distance from the diagonal if any
    min_dist_drop_mask = (
        all_pattern_coords.chrom1 == all_pattern_coords.chrom2
    ) & (
        np.abs(all_pattern_coords.start2 - all_pattern_coords.start1)
        < cfg["min_dist"]
    )
    # Reorder columns at the same time
    all_pattern_coords = all_pattern_coords.loc[
        ~min_dist_drop_mask,
        [
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            "bin1",
            "bin2",
            "kernel_id",
            "iteration",
            "score",
        ],
    ]
    all_pattern_windows = all_pattern_windows[~min_dist_drop_mask, :, :]

    ### 3: WRITE OUTPUT
    sys.stderr.write(f"{all_pattern_coords.shape[0]} patterns detected\n")
    # Save patterns and their coordinates in a tsv file
    cio.write_patterns(
        all_pattern_coords, cfg["name"] + "_out", output
    )
    # Save windows as an array in an npy file
    cio.save_windows(
        all_pattern_windows,
        cfg["name"] + "_out",
        output,
        format=win_fmt,
    )

    # Generate pileup visualisations if requested
    if plotting_enabled:
        # Compute and plot pileup
        pileup_fname = ("pileup_of_{n}_{pattern}").format(
            pattern=cfg["name"], n=all_pattern_windows.shape[0]
        )
        windows_pileup = cid.pileup_patterns(all_pattern_windows)
        pileup_plot(windows_pileup, name=pileup_fname, output=output)


def cmd_test(arguments):

    sys.stderr.write(f"Fetching test dataset at {URL_EXAMPLE_DATASET}...\n")
    test_data = pd.read_csv(URL_EXAMPLE_DATASET, sep="\t")

    # Turn dataframe into file like object for the detector to parse
    test_stream = io.StringIO()
    test_data.to_csv(test_stream)
    test_stream.seek(0)

    sys.stderr.write(f"Running detection on test dataset...\n")

    arguments["<contact_map>"] = test_stream
    cmd_detect(arguments)


@contextmanager
def capture_ouput(stderr_to=None):
    """Capture the stderr of the test run. Inspired from
    http://sametmax.com/capturer-laffichage-des-prints-dun-code-python/
    """

    try:
        stderr = sys.stderr
        sys.stderr = c2 = stderr_to or io.StringIO()
        yield c2

    finally:
        sys.stderr = stderr
        try:
            c2.flush()
            c2.seek(0)
        except (ValueError, IOError):
            pass


def main():
    arguments = docopt.docopt(__doc__, version=__version__)
    detect = arguments["detect"]
    generate_config = arguments["generate-config"]
    quantify = arguments["quantify"]
    test = arguments["test"]
    if test:
        with capture_ouput() as stderr:
            cmd_test(arguments)

        actual_log = stderr.read()
        sys.stderr.write(actual_log)

        # remove progress bars and \r chars
        actual_log_lines = {
            u.strip("\r") for u in set(actual_log.split("\n")) if "[" not in u
        }
        expected_log_lines = set(TEST_LOG.split("\n"))

        if expected_log_lines not in actual_log_lines:
            sys.stderr.write(
                "\nWarning, the test log differed from the "
                "expected one. This means the program changed its output from"
                "previous versions. You may ignore this if you are not a "
                "developer.\n\n"
                f"Here is the expected log:\n\n{TEST_LOG}\n"
            )

    elif detect:
        cmd_detect(arguments)
    elif generate_config:
        cmd_generate_config(arguments)
    elif quantify:
        cmd_quantify(arguments)
    return 0


if __name__ == "__main__":
    main()
