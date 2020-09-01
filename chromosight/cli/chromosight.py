#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops]
                        [--pearson=auto] [--win-size=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--force-norm]
                        [--subsample=no] [--inter] [--tsvd] [--smooth-trend]
                        [--n-mads=5] [--min-dist=0] [--max-dist=auto]
                        [--no-plotting] [--min-separation=auto] [--dump=DIR]
                        [--threads=1] [--perc-zero=auto]
                        [--perc-undetected=auto] <contact_map> <prefix>
    chromosight generate-config [--preset loops] [--click contact_map]
                        [--force-norm] [--win-size=auto] [--n-mads=5]
                        [--threads=1] <prefix>
    chromosight quantify [--inter] [--pattern=loops] [--subsample=no]
                         [--win-fmt=json] [--kernel-config=FILE] [--force-norm]
                         [--threads=1] [--n-mads=5] [--win-size=auto]
                         [--perc-undetected=auto] [--perc-zero=auto]
                         [--no-plotting] [--tsvd] <bed2d> <contact_map> <prefix>
    chromosight list-kernels [--long] [--mat] [--name=kernel_name]
    chromosight test

    detect:
        performs pattern detection on a Hi-C contact map via template matching
    generate-config:
        Generate pre-filled config files to use for detect and quantify.
        A config consists of a JSON file describing parameters for the
        analysis and path pointing to kernel matrices files. Those matrices
        files are tsv files with numeric values as kernel to use for
        convolution.
    quantify:
        Given a list of pairs of positions and a contact map, computes the
        correlation coefficients between those positions and the kernel of the
        selected pattern.
    list-kernels:
        Prints information about available kernels.
    test:
        Download example data and run loop detection on it.

Arguments for detect:
    contact_map                 The Hi-C contact map to detect patterns on, in
                                bedgraph2d or cool format.
    prefix                      Common path prefix used to generate output files.
                                Extensions will be added for each file.

Arguments for quantify:
    bed2d                       Tab-separated text files with columns chrom1, start1
                                end1, chrom2, start2, end2. Each line correspond to
                                a pair of positions (i.e. a position in the matrix).
    contact_map                 Path to the contact map, in bedgraph2d or
                                cool format.
    prefix                      Common path prefix used to generate output files.
                                Extensions will be added for each file.

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

Arguments for list-kernels:
    --name=kernel_name      Only show information related to a particular
                            kernel.[default: all]
    --long                  Show default parameters in addition to kernel names.
    --mat                   Prints an ascii representation of the kernel matrix.

Basic options:
    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    --verbose                   Displays the logo.
    -F, --force-norm            Re-compute matrix normalization (balancing) and
                                overwrite weights present in the cool files instead
                                of reusing them.
    -I, --inter                 Enable to consider interchromosomal contacts.
                                Warning: Experimental feature with high memory
                                consumption, only use with small matrices.
    -m, --min-dist=auto         Minimum distance from the diagonal (in base pairs).
                                at which detection should operate. [default: auto]
    -M, --max-dist=auto         Maximum distance from the diagonal (in base pairs)
                                for detection. [default: auto]
    -P, --pattern=loops         Which pattern to detect. This will use preset
                                configurations for the given pattern. Possible
                                values are: loops, borders, hairpins and
                                centromeres. [default: loops]
    -p, --pearson=auto          Pearson correlation threshold when detecting patterns
                                in the contact map. Lower values leads to potentially
                                more detections, but more false positives. [default: auto]
    -s, --subsample=INT         If greater than 1, subsample INT contacts from the
                                matrix. If between 0 and 1, subsample a proportion of
                                contacts instead. Useful when comparing matrices with
                                different coverages. [default: no]
    -t, --threads=1             Number of CPUs to use in parallel. [default: 1]
    -u, --perc-undetected=auto  Maximum percentage of non-detectable pixels (nan) in
                                windows allowed to report patterns. [default: auto]
    -z, --perc-zero=auto        Maximum percentage of empty (0) pixels in windows
                                allowed to report patterns. [default: auto]

Advanced options:
    -d, --dump=DIR              Directory where to save matrix dumps during
                                processing and detection. Each dump is saved as
                                a compressed npz of a sparse matrix and can be
                                loaded using scipy.sparse.load_npz.
    -i, --iterations=auto       How many iterations to perform after the first
                                template-based pass. [default: 1]
    -k, --kernel-config=FILE    Optionally give a path to a custom JSON kernel
                                config path. Use this to override pattern if
                                you do not want to use one of the preset
                                patterns.
    -n, --no-plotting           Disable generation of pileup plots.
    -N, --n-mads=5              Maximum number of median absolute deviations below
                                the median of the bin sums distribution allowed to
                                consider detectable bins. [default: 5]
    -S, --min-separation=auto   Minimum distance required between patterns, in
                                basepairs. If two patterns are closer than this
                                distance in both axes, the one with the lowest
                                score is discarded. [default: auto]
    -T, --smooth-trend          Use isotonic regression when detrending to reduce
                                noise at long ranges. Do not enable this for circular
                                genomes.
    -V, --tsvd                  Enable kernel factorisation via truncated svd.
                                Accelerates detection, at the cost of slight
                                inaccuracies. Singular matrices are truncated to
                                retain 99.9% of the information in the kernel.
    -w, --win-fmt={json,npy}    File format used to store individual windows
                                around each pattern. Window order matches
                                patterns inside the associated text file.
                                Possible formats are json and npy. [default: json]
    -W, --win-size=auto         Window size (width), in pixels, to use for the
                                kernel when computing correlations. The pattern
                                kernel will be resized to match this size. Linear
                                linear interpolation is used to fill between pixels.
                                If not specified, the default kernel size will
                                be used instead. [default: auto]


"""
import numpy as np
import pandas as pd
import os
import io
from contextlib import contextmanager
import sys
import json
import docopt
import tempfile
import multiprocessing as mp
from chromosight.version import __version__
from chromosight.utils.contacts_map import HicGenome
import chromosight.utils.io as cio
import chromosight.utils.detection as cid
from chromosight.utils.plotting import pileup_plot, click_finder, print_ascii_mat
from chromosight.utils.preprocessing import resize_kernel
from chromosight.utils.stats import fdr_correction
import chromosight.kernels as ck
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pathlib
LOGO = np.loadtxt(pathlib.Path(__file__).parents[0] / 'logo.txt')
URL_EXAMPLE_DATASET = (
    "https://raw.githubusercontent.com/koszullab/"
    "chromosight/master/data_test/example.cool"
)

TEST_LOG = f"""Fetching test dataset at {URL_EXAMPLE_DATASET}...
Running detection on test dataset...
pearson set to 0.3 based on config file.
max_dist set to 2000000 based on config file.
min_dist set to 20000 based on config file.
min_separation set to 5000 based on config file.
max_perc_undetected set to 50.0 based on config file.
max_perc_zero set to 10.0 based on config file.
Matrix already balanced, reusing weights
Preprocessing sub-matrices...
Detecting patterns...
89 patterns detected
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


def cmd_quantify(args):
    bed2d_path = args["<bed2d>"]
    mat_path = args["<contact_map>"]
    prefix = args["<prefix>"]
    n_mads = float(args["--n-mads"])
    pattern = args["--pattern"]
    inter = args["--inter"]
    kernel_config_path = args["--kernel-config"]
    perc_zero = args["--perc-zero"]
    perc_undetected = args["--perc-undetected"]
    plotting_enabled = False if args["--no-plotting"] else True
    threads = int(args["--threads"])
    force_norm = args["--force-norm"]
    tsvd = 0.999 if args["--tsvd"] else None
    win_fmt = args["--win-fmt"]
    if win_fmt not in ["npy", "json"]:
        sys.stderr.write("Error: --win-fmt must be either json or npy.\n")
        sys.exit(1)
    win_size = args["--win-size"]
    if win_size != "auto":
        win_size = int(win_size)
    subsample = args["--subsample"]
    # If prefix involves a directory, crash if it does not exist
    cio.check_prefix_dir(prefix)
    # Load 6 cols from 2D BED file and infer header
    bed2d = cio.load_bed2d(bed2d_path)
    # Warn user if --inter is disabled but list contains inter patterns
    if not inter and len(bed2d.start1[bed2d.chrom1 != bed2d.chrom2]) > 0:
        sys.stderr.write(
            "Warning: The bed2d file contains interchromosomal patterns. "
            "These patterns will not be scanned unless --inter is used.\n"
        )
    if kernel_config_path is not None:
        custom = True
        # Loading input path as config
        config_path = kernel_config_path
    else:
        custom = False
        # Will use a preset config file matching pattern name
        config_path = pattern
    cfg = cio.load_kernel_config(config_path, custom)
    # Subsample Hi-C contacts from the matrix, if requested
    if subsample == "no":
        subsample = None
    # Instantiate and preprocess contact map
    hic_genome = HicGenome(
        mat_path, inter=inter, kernel_config=cfg, sample=subsample
    )
    # enforce max scanning distance to pattern at longest distance
    furthest = np.max(bed2d.start2 - bed2d.start1)
    max_diag = hic_genome.clr.shape[0] * hic_genome.clr.binsize
    cfg["max_dist"] = min(furthest, max_diag)
    cfg["min_dist"] = 0
    cfg = _override_kernel_config("max_perc_zero", perc_zero, float, cfg)
    cfg = _override_kernel_config(
        "max_perc_undetected", perc_undetected, float, cfg
    )

    # Notify contact map instance of changes in scanning distance
    hic_genome.kernel_config = cfg
    # Normalize (balance) matrix using ICE
    hic_genome.normalize(force_norm=force_norm, n_mads=n_mads, threads=threads)
    # Initialize output structures
    bed2d["score"] = np.nan
    bed2d["pvalue"] = np.nan
    positions = bed2d.copy()
    # Only resize kernel matrix if explicitely requested
    km, kn = cfg["kernels"][0].shape
    if win_size != "auto":
        if not win_size % 2:
            raise ValueError("--win-size must be odd")
        for i, k in enumerate(cfg["kernels"]):
            cfg["kernels"][i] = resize_kernel(k, factor=win_size / km)
        km = kn = win_size
        # Update kernel config after resizing kernels
        hic_genome.kernel_config = cfg
    # Define how many diagonals should be used in intra-matrices
    hic_genome.compute_max_dist()
    # Split whole genome matrix into intra- and inter- sub matrices. Each sub
    # matrix is processed on the fly (obs / exp, trimming diagonals > max dist)
    hic_genome.make_sub_matrices()
    windows = np.full((positions.shape[0], km, kn), np.nan)
    # For each position, we use the center of the BED interval
    positions["pos1"] = (positions.start1 + positions.end1) // 2
    positions["pos2"] = (positions.start2 + positions.end2) // 2
    # Use each kernel matrix available for the pattern
    for kernel_id, kernel_matrix in enumerate(cfg["kernels"]):
        cio.progress(kernel_id, len(cfg["kernels"]), f"Kernel: {kernel_id}\n")
        # Iterate over intra- and inter-chromosomal sub-matrices
        n_sub_mats = hic_genome.sub_mats.shape[0]
        for sub_mat_id, sub_mat in enumerate(hic_genome.sub_mats.iterrows()):
            mat = sub_mat[1]
            cio.progress(sub_mat_id, n_sub_mats, f"{mat.chr1}-{mat.chr2}")
            mat.contact_map.create_mat()
            # Filter patterns falling onto this sub-matrix
            sub_pat = positions.loc[
                (positions.chrom1 == mat.chr1) & (positions.chrom2 == mat.chr2)
            ]
            # Convert genomic coordinates to bins for horizontal and vertical axes
            for ax in [1, 2]:
                sub_pat_ax = sub_pat.loc[:, [f"chrom{ax}", f"pos{ax}"]].rename(
                    columns={f"chrom{ax}": "chrom", f"pos{ax}": "pos"}
                )
                sub_pat_bins = hic_genome.coords_to_bins(sub_pat_ax)
                sub_pat[f"bin{ax}"] = sub_pat_bins

            # Check for nan bins (coords that do not match any Hi-C fragments
            fall_out = np.isnan(sub_pat["bin1"]) | np.isnan(sub_pat["bin2"])
            if np.any(fall_out):
                n_out = len(sub_pat_bins[fall_out])
                sys.stderr.write(
                    f"\n{n_out} entr{'ies' if n_out > 1 else 'y'} outside "
                    "genomic coordinates of the Hi-C matrix will be ignored.\n"
                )
                sub_pat = sub_pat.loc[~fall_out, :]
            sub_pat_idx = sub_pat.index.values
            # Convert bins from whole genome matrix to sub matrix
            sub_pat = hic_genome.get_sub_mat_pattern(
                mat.chr1, mat.chr2, sub_pat
            )
            m = mat.contact_map.matrix.tocsr()

            # Feed the submatrix to quantification pipeline
            patterns, mat_windows = cid.pattern_detector(
                mat.contact_map,
                cfg,
                kernel_matrix,
                coords=np.array(sub_pat.loc[:, ["bin1", "bin2"]]),
                full=True,
                tsvd=tsvd,
            )

            # For each coordinate, keep the highest coefficient
            # among all kernels.
            try:
                if kernel_id == 0:
                    bed2d["score"][sub_pat_idx] = patterns.score.values
                    bed2d["pvalue"][sub_pat_idx] = patterns.pvalue.values
                    windows[sub_pat_idx, :, :] = mat_windows
                else:
                    # Only update scores and their corresponding windows
                    # if better than results from previous kernels
                    better = (
                        patterns.score > bed2d["score"][sub_pat_idx].values
                    ) | (np.insnan(bed2d["score"][sub_pat_idx].values))
                    better_idx = sub_pat_idx[better]
                    bed2d["score"][better_idx] = patterns.score[better].values
                    windows[better_idx, :, :] = mat_windows[better]
            # Do nothing if no pattern was detected or matrix
            # is smaller than the kernel (-> patterns is None)
            except AttributeError:
                pass
            # Free space from current submatrix
            mat.contact_map.destroy_mat()
            del m
            m = None
        bed2d["bin1"] = hic_genome.coords_to_bins(
            bed2d.loc[:, ["chrom1", "start1"]].rename(
                columns={"chrom1": "chrom", "start1": "pos"}
            )
        )
        bed2d["bin2"] = hic_genome.coords_to_bins(
            bed2d.loc[:, ["chrom2", "start2"]].rename(
                columns={"chrom2": "chrom", "start2": "pos"}
            )
        )
        bed2d["qvalue"] = fdr_correction(bed2d["pvalue"])
        bed2d = bed2d.loc[
            :,
            [
                "chrom1",
                "start1",
                "end1",
                "chrom2",
                "start2",
                "end2",
                "bin1",
                "bin2",
                "score",
                "pvalue",
                "qvalue",
            ],
        ]
        # Set p-values of invalid scores to nan
        bed2d.loc[np.isnan(bed2d.score), "pvalue"] = np.nan
        bed2d.loc[np.isnan(bed2d.score), "qvalue"] = np.nan
        cio.write_patterns(bed2d, prefix)
        cio.save_windows(windows, prefix, fmt=win_fmt)
        # Generate pileup visualisations if requested
        if plotting_enabled:
            # Compute and plot pileup
            pileup_title = ("pileup_of_{n}_{pattern}").format(
                pattern=cfg["name"], n=windows.shape[0]
            )
            windows_pileup = cid.pileup_patterns(windows)
            # Symmetrize pileup for diagonal patterns
            if not cfg["max_dist"]:
                # Replace nan below diag by 0
                windows_pileup = np.nan_to_num(windows_pileup)
                # Add transpose
                windows_pileup += np.transpose(windows_pileup) - np.diag(
                    np.diag(windows_pileup)
                )
            pileup_plot(windows_pileup, prefix, name=pileup_title)


def cmd_generate_config(args):
    # Parse command line args for generate_config
    prefix = args["<prefix>"]
    pattern = args["--preset"]
    click_find = args["--click"]
    n_mads = float(args["--n-mads"])
    force_norm = args["--force-norm"]
    win_size = args["--win-size"]
    threads = int(args["--threads"])

    cfg = cio.load_kernel_config(pattern, False)

    # If prefix involves a directory, crash if it does not exist
    cio.check_prefix_dir(prefix)

    # If a specific window size if requested, resize all kernels
    if win_size != "auto":
        win_size = int(win_size)
        if not win_size % 2:
            raise ValueError("--win-size must be odd")
        resize = lambda m: resize_kernel(m, factor=win_size / m.shape[0])
        cfg["kernels"] = [resize(k) for k in cfg["kernels"]]
    # Otherwise, just inherit window size from the kernel config
    else:
        win_size = cfg["kernels"][0].shape[0]

    # If click mode is enabled, build a kernel from scratch using
    # graphical display, otherwise, just inherit the pattern's kernel
    if click_find:
        hic_genome = HicGenome(click_find, inter=True, kernel_config=cfg)
        # Normalize (balance) the whole genome matrix
        hic_genome.normalize(
            force_norm=force_norm, n_mads=n_mads, threads=threads
        )
        # enforce full scanning distance in kernel config

        hic_genome.max_dist = hic_genome.clr.shape[0] * hic_genome.clr.binsize
        # Process each sub-matrix individually (detrend diag for intra)
        hic_genome.make_sub_matrices()
        for sub in hic_genome.sub_mats.iterrows():
            sub_mat = sub[1]
            sub_mat.contact_map.create_mat()
        processed_mat = hic_genome.gather_sub_matrices().tocsr()
        windows = click_finder(processed_mat, half_w=int((win_size - 1) / 2))
        # Pileup all recorded windows and convert to JSON serializable list
        pileup = ndi.gaussian_filter(cid.pileup_patterns(windows), 1)
        cfg["kernels"] = [pileup.tolist()]
        # Show the newly generate kernel to the user, use zscore to highlight contrast
        hm = plt.imshow(
            np.log(pileup), vmax=np.percentile(pileup, 99), cmap="afmhot_r"
        )
        cbar = plt.colorbar(hm)
        cbar.set_label("Log10 Hi-C contacts")
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
    sub.contact_map.create_mat()
    chrom_patterns, chrom_windows = cid.pattern_detector(
        sub.contact_map,
        config,
        kernel,
        dump=dump,
        full=True,
        tsvd=config["tsvd"],
    )
    sub.contact_map.destroy_mat()

    return {
        "coords": chrom_patterns,
        "windows": chrom_windows,
        "chr1": sub.chr1,
        "chr2": sub.chr2,
    }


def cmd_detect(args):
    # Parse command line arguments for detect
    dump = args["--dump"]
    force_norm = args["--force-norm"]
    interchrom = args["--inter"]
    iterations = args["--iterations"]
    kernel_config_path = args["--kernel-config"]
    mat_path = args["<contact_map>"]
    max_dist = args["--max-dist"]
    min_dist = args["--min-dist"]
    min_separation = args["--min-separation"]
    n_mads = float(args["--n-mads"])
    prefix = args["<prefix>"]
    pattern = args["--pattern"]
    pearson = args["--pearson"]
    perc_zero = args["--perc-zero"]
    perc_undetected = args["--perc-undetected"]
    subsample = args["--subsample"]
    threads = int(args["--threads"])
    tsvd = 0.999 if args["--tsvd"] else None
    win_fmt = args["--win-fmt"]
    win_size = args["--win-size"]
    if subsample == "no":
        subsample = None
    plotting_enabled = False if args["--no-plotting"] else True
    smooth_trend = args["--smooth-trend"]
    if smooth_trend is None:
        smooth_trend = False

    # If prefix involves a directory, crash if it does not exist
    cio.check_prefix_dir(prefix)

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
        "pearson": (pearson, float),
        "max_dist": (max_dist, int),
        "min_dist": (min_dist, int),
        "min_separation": (min_separation, int),
        "max_perc_undetected": (perc_undetected, float),
        "max_perc_zero": (perc_zero, float),
    }
    cfg = cio.load_kernel_config(config_path, custom)
    for param_name, (param_value, param_type) in params.items():
        cfg = _override_kernel_config(param_name, param_value, param_type, cfg)

    # Resize kernels if requested
    if win_size != "auto":
        win_size = int(win_size)
        if not win_size % 2:
            raise ValueError("--win-size must be odd")
        resize = lambda m: resize_kernel(m, factor=win_size / m.shape[0])
        cfg["kernels"] = [resize(k) for k in cfg["kernels"]]

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
        sample=subsample,
    )
    ### 1: Process input signal
    hic_genome.kernel_config = cfg
    # Normalize (balance) matrix using ICE
    hic_genome.normalize(force_norm=force_norm, n_mads=n_mads, threads=threads)
    # Define how many diagonals should be used in intra-matrices
    hic_genome.compute_max_dist()
    # Split whole genome matrix into intra- and inter- sub matrices. Each sub
    # matrix is processed on the fly (obs / exp, trimming diagonals > max dist)
    hic_genome.make_sub_matrices()

    all_coords = []
    all_windows = []

    ### 2: DETECTION ON EACH SUBMATRIX
    n_sub_mats = hic_genome.sub_mats.shape[0]
    # Loop over the different kernel matrices for input pattern
    run_id = 0
    # Use cfg to inform jobs whether they should run full convolution
    cfg["tsvd"] = tsvd
    total_runs = len(cfg["kernels"]) * cfg["max_iterations"]
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
            # Run in multiprocessing subprocesses
            if threads > 1:
                pool = mp.Pool(threads)
                dispatcher = pool.imap(_detect_sub_mat, sub_mat_data, 1)
            else:
                dispatcher = map(_detect_sub_mat, sub_mat_data)
            for s, result in enumerate(dispatcher):
                chr1 = hic_genome.sub_mats.chr1[s]
                chr2 = hic_genome.sub_mats.chr2[s]
                cio.progress(s, n_sub_mats, f"{chr1}-{chr2}")
                sub_mat_results.append(result)

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
                all_coords.append(
                    pd.concat(kernel_coords, axis=0).reset_index(drop=True)
                )
                # Add info about kernel and iteration which detected these patterns
                all_coords[-1]["kernel_id"] = kernel_id
                all_coords[-1]["iteration"] = i
                all_windows.append(kernel_windows)

            # If no pattern was found with this kernel
            # skip directly to the next one, skipping iterations
            except ValueError:
                break

            # Update kernel with patterns detected at current iteration
            kernel_matrix = cid.pileup_patterns(kernel_windows)
            run_id += 1
    cio.progress(run_id, total_runs, f"Kernel: {kernel_id}, Iteration: {i}\n")
    # If no pattern detected on any chromosome, with any kernel, exit gracefully
    if len(all_coords) == 0:
        sys.stderr.write("No pattern detected ! Exiting.\n")
        sys.exit(0)
    # Finish parallelized part
    if threads > 1:
        pool.close()
    # Combine patterns of all kernel matrices into a single array
    all_coords = pd.concat(all_coords, axis=0).reset_index(drop=True)
    # Combine all windows from different kernels into a single pile of windows
    all_windows = np.concatenate(all_windows, axis=0)

    # Compute minimum separation in bins and make sure it has a reasonable value
    separation_bins = int(cfg["min_separation"] // hic_genome.clr.binsize)
    if separation_bins < 1:
        separation_bins = 1
    print(f"Minimum pattern separation is : {separation_bins}")
    # Remove patterns with overlapping windows (smeared patterns)
    distinct_patterns = cid.remove_neighbours(
        all_coords, win_size=separation_bins
    )

    # Drop patterns that are too close to each other
    all_coords = all_coords.loc[distinct_patterns, :]
    all_windows = all_windows[distinct_patterns, :, :]

    # Get from bins into basepair coordinates
    coords_1 = hic_genome.bins_to_coords(all_coords.bin1).reset_index(
        drop=True
    )
    coords_1.columns = [str(col) + "1" for col in coords_1.columns]
    coords_2 = hic_genome.bins_to_coords(all_coords.bin2).reset_index(
        drop=True
    )
    coords_2.columns = [str(col) + "2" for col in coords_2.columns]

    all_coords = pd.concat(
        [all_coords.reset_index(drop=True), coords_1, coords_2], axis=1
    )

    # Filter patterns closer than minimum distance from the diagonal if any
    min_dist_drop_mask = (all_coords.chrom1 == all_coords.chrom2) & (
        np.abs(all_coords.start2 - all_coords.start1) < cfg["min_dist"]
    )
    all_coords = all_coords.loc[~min_dist_drop_mask, :]
    all_windows = all_windows[~min_dist_drop_mask, :, :]

    # Correct p-values for multiple testing using FDR
    all_coords["qvalue"] = fdr_correction(all_coords["pvalue"])
    # Reorder columns
    all_coords = all_coords.loc[
        :,
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
            "pvalue",
            "qvalue",
        ],
    ]

    ### 3: WRITE OUTPUT
    sys.stderr.write(f"{all_coords.shape[0]} patterns detected\n")
    # Save patterns and their coordinates in a tsv file
    cio.write_patterns(all_coords, prefix)
    # Save windows as an array in an npy file
    cio.save_windows(all_windows, prefix, fmt=win_fmt)

    # Generate pileup visualisations if requested
    if plotting_enabled:
        # Compute and plot pileup
        pileup_title = ("Pileup of {n} {pattern}").format(
            pattern=cfg["name"], n=all_windows.shape[0]
        )
        windows_pileup = cid.pileup_patterns(all_windows)
        # Symmetrize pileup for diagonal patterns
        if not cfg["max_dist"]:
            # Replace nan below diag by 0
            windows_pileup = np.nan_to_num(windows_pileup)
            # Add transpose
            windows_pileup += np.transpose(windows_pileup) - np.diag(
                np.diag(windows_pileup)
            )
        pileup_plot(windows_pileup, prefix, name=pileup_title)



def cmd_list_kernels(args):

    kernel_name = args["--name"]
    # Load every avaiable kernel by default
    if kernel_name == 'all':
        kernels = ck.kernel_names
    # If a specific kernel was requested, only load this one
    else:
        kernels = [kernel_name]
    
    # Check availability of each kernel and print its name
    for k in kernels:
        try:
            kernel_infos = getattr(ck, k)
        except AttributeError:
            raise ValueError(f"Kernel {k} is not available")
        print(k)
        # Print default params if --long specified (key-value pairs in json)
        if args['--long']:
            exclude_params = ['name', 'resolution', 'kernels']
            for param, value in kernel_infos.items():
                if param not in exclude_params:
                    print(f"  {param}: {value}")
        if args['--mat']:
            mats = kernel_infos['kernels']
            for mat in mats:
                print_ascii_mat(mat)



def cmd_test(args):

    sys.stderr.write(f"Fetching test dataset at {URL_EXAMPLE_DATASET}...\n")
    tmp_cool = tempfile.NamedTemporaryFile(delete=False)
    cio.download_file(URL_EXAMPLE_DATASET, tmp_cool.name)

    sys.stderr.write(f"Running detection on test dataset...\n")

    args["<contact_map>"] = tmp_cool.name
    args["<prefix>"] = 'chromosight_test'
    args["--no-plotting"] = True
    cmd_detect(args)
    os.unlink(tmp_cool.name)


@contextmanager
def capture_ouput(stderr_to=None):
    """Capture the stderr of the test run. """

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

def logo_version(logo, ver):
    small_logo = resize_kernel(logo, factor=.33, quiet=True)
    ascii_logo = print_ascii_mat(small_logo, colored=False, print_str=False)
    return f'{ascii_logo} Chromosight version {ver}'

def main():

    args = docopt.docopt( __doc__, version=logo_version(LOGO, __version__))
    detect = args["detect"]
    generate_config = args["generate-config"]
    list_kernels = args["list-kernels"]
    quantify = args["quantify"]
    test = args["test"]
    if test:
        with capture_ouput() as stderr:
            cmd_test(args)

        obs_log = stderr.read()
        sys.stderr.write(obs_log)

        # remove progress bars and \r chars
        obs_log_lines = {
            u.strip("\x1b[K")
            for u in set(obs_log.split("\n"))
            if "\r" not in u
        }
        exp_log_lines = set(TEST_LOG.split("\n"))

        if len(exp_log_lines ^ obs_log_lines):
            sys.stderr.write(
                "\nWarning, the test log differed from the "
                "expected one. This means the program changed its output from"
                "previous versions. You may ignore this if you are not a "
                "developer.\n\n"
                f"Here is the expected log:\n\n{TEST_LOG}\n"
            )

    elif detect:
        cmd_detect(args)
    elif generate_config:
        cmd_generate_config(args)
    elif list_kernels:
        cmd_list_kernels(args)
    elif quantify:
        cmd_quantify(args)
        
    return 0


if __name__ == "__main__":
    main()
