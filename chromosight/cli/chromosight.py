#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops]
                        [--pearson=auto] [--win-size=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--force-norm] [--full]
                        [--subsample=no] [--inter] [--tsvd] [--smooth-trend]
                        [--n-mads=5] [--min-dist=0] [--max-dist=auto]
                        [--no-plotting] [--min-separation=auto] [--dump=DIR]
                        [--threads=1] [--perc-undetected=auto] <contact_map>
                        [<output>]
    chromosight generate-config [--preset loops] [--click contact_map]
                        [--force-norm] [--win-size=auto] [--n-mads=5]
                        [--threads=1] <prefix>
    chromosight quantify [--inter] [--pattern=loops] [--subsample=no]
                         [--win-fmt=json] [--kernel-config=FILE] [--force-norm]
                         [--threads=1] [--full] [--n-mads=5] [--win-size=auto]
                         [--tsvd] <bed2d> <contact_map> <output>
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
    test:
        Download example data and run loop detection on it.

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
    -f, --full                  Enable 'full' convolution mode: The whole matrix
                                is scanned all the way to edges and missing bins
                                are masked. This will allow to detect very close
                                to the diagonal and close to repeated sequences
                                at the cost of memory and compute time.
    -F, --force-norm            Re-compute matrix normalization (balancing) and
                                overwrite weights present in the cool files instead
                                of reusing them.
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
    -p, --pearson=auto          Pearson correlation threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: auto]
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
    -W, --win-size=auto         Window size (width), in pixels, to use for the
                                kernel when computing correlations. The pattern
                                kernel will be resized to match this size. If
                                the pattern must be enlarged, linear
                                interpolation is used to fill between pixels.
                                If not specified, the default kernel size will
                                be used instead. [default: auto]
    -V, --tsvd                  Enable kernel factorisation via truncated svd.
                                This should accelerate detection in most cases,
                                at the cost of slight inaccuracies. The singular
                                matrices are truncated so that 99.9% of the
                                information contained in the kernel is retained.

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
import tempfile
import multiprocessing as mp
from chromosight.version import __version__
from chromosight.utils.contacts_map import HicGenome
import chromosight.utils.io as cio
import chromosight.utils.detection as cid
from chromosight.utils.plotting import pileup_plot, click_finder
from chromosight.utils.preprocessing import resize_kernel
from chromosight.utils.stats import fdr_correction
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

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
max_perc_undetected set to 10.0 based on config file.
Matrix already balanced, reusing weights
Preprocessing sub-matrices...
Detecting patterns...
21 patterns detected
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
    full = args["--full"]
    output = pathlib.Path(args["<output>"])
    n_mads = float(args["--n-mads"])
    pattern = args["--pattern"]
    inter = args["--inter"]
    kernel_config_path = args["--kernel-config"]
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
    # Notify contact map instance of changes in scanning distance
    hic_genome.kernel_config = cfg
    # Normalize (balance) matrix using ICE
    hic_genome.normalize(force_norm=force_norm, n_mads=n_mads, threads=threads)
    # Define how many diagonals should be used in intra-matrices
    hic_genome.compute_max_dist()
    # Split whole genome matrix into intra- and inter- sub matrices. Each sub
    # matrix is processed on the fly (obs / exp, trimming diagonals > max dist)
    hic_genome.make_sub_matrices()
    # Initialize output structures
    bed2d["score"] = np.nan
    bed2d["pvalue"] = np.nan
    positions = bed2d.copy()
    if win_size != "auto":
        km = kn = win_size
    else:
        km, kn = cfg["kernels"][0].shape
    windows = np.full((positions.shape[0], km, kn), np.nan)
    # For each position, we use the center of the BED interval
    positions["pos1"] = (positions.start1 + positions.end1) // 2
    positions["pos2"] = (positions.start2 + positions.end2) // 2
    # Use each kernel matrix available for the pattern
    for kernel_id, kernel_matrix in enumerate(cfg["kernels"]):
        cio.progress(kernel_id, len(cfg["kernels"]), f"Kernel: {kernel_id}\n")
        # Only resize kernel matrix if explicitely requested
        if win_size != "auto":
            kernel_matrix = resize_kernel(kernel_matrix, factor=win_size / km)
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
            sub_pat_idx = sub_pat.index.values
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
                    f"{n_out} entr{'ies' if n_out > 1 else 'y'} outside "
                    "genomic coordinates of the Hi-C matrix will be ignored.\n"
                )
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
                full=full,
                tsvd=tsvd,
            )

            # For each coordinate, keep the highest coefficient
            # among all kernels.
            try:
                if kernel_id == 0:
                    bed2d["score"][sub_pat_idx] = patterns.score
                    bed2d["pvalue"][sub_pat_idx] = patterns.pvalue
                    windows[sub_pat_idx, :, :] = mat_windows
                else:
                    # Only update scores and their corresponding windows
                    # if better than results from previous kernels
                    better = (
                        patterns.score > bed2d["score"][sub_pat_idx].values
                    )
                    better_idx = sub_pat_idx[better]
                    bed2d["score"][better_idx] = patterns.score[better]
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
        cio.write_patterns(bed2d, f"{pattern}_quant", output)
        # bed2d.to_csv(
        #    output / f"{pattern}_quant.txt", sep="\t", header=True, index=False
        # )
        cio.save_windows(
            windows, f"{pattern}_quant", output_dir=output, format=win_fmt
        )


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

    # If prefix involves a directory, create it
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # If a specific window size if requested, resize all kernels
    if win_size != "auto":
        win_size = int(win_size)
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
        full=config["full"],
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
    full = args["--full"]
    interchrom = args["--inter"]
    iterations = args["--iterations"]
    kernel_config_path = args["--kernel-config"]
    mat_path = args["<contact_map>"]
    max_dist = args["--max-dist"]
    min_dist = args["--min-dist"]
    min_separation = args["--min-separation"]
    n_mads = float(args["--n-mads"])
    output = args["<output>"]
    pattern = args["--pattern"]
    pearson = args["--pearson"]
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
        "pearson": (pearson, float),
        "max_dist": (max_dist, int),
        "min_dist": (min_dist, int),
        "min_separation": (min_separation, int),
        "max_perc_undetected": (perc_undetected, float),
    }
    cfg = cio.load_kernel_config(config_path, custom)
    for param_name, (param_value, param_type) in params.items():
        cfg = _override_kernel_config(param_name, param_value, param_type, cfg)

    # Resize kernels if requested
    if win_size != "auto":
        resize = lambda m: resize_kernel(m, factor=int(win_size) / m.shape[0])
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

    all_pattern_coords = []
    all_pattern_windows = []

    ### 2: DETECTION ON EACH SUBMATRIX
    pool = mp.Pool(threads)
    n_sub_mats = hic_genome.sub_mats.shape[0]
    # Loop over the different kernel matrices for input pattern
    run_id = 0
    # Use cfg to inform jobs whether they should run full convolution
    cfg["full"] = full
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
                dispatcher = pool.imap_unordered(
                    _detect_sub_mat, sub_mat_data, 1
                )
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
    separation_bins = int(cfg["min_separation"] // hic_genome.clr.binsize)
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
    all_pattern_coords = all_pattern_coords.loc[~min_dist_drop_mask, :]
    all_pattern_windows = all_pattern_windows[~min_dist_drop_mask, :, :]
    # Correct p-values for multiple testing using FDR
    all_pattern_coords["qvalue"] = fdr_correction(all_pattern_coords["pvalue"])
    # Reorder columns
    all_pattern_coords = all_pattern_coords.loc[
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
    sys.stderr.write(f"{all_pattern_coords.shape[0]} patterns detected\n")
    # Save patterns and their coordinates in a tsv file
    cio.write_patterns(all_pattern_coords, cfg["name"] + "_out", output)
    # Save windows as an array in an npy file
    cio.save_windows(
        all_pattern_windows, cfg["name"] + "_out", output, format=win_fmt
    )

    # Generate pileup visualisations if requested
    if plotting_enabled:
        # Compute and plot pileup
        pileup_fname = ("pileup_of_{n}_{pattern}").format(
            pattern=cfg["name"], n=all_pattern_windows.shape[0]
        )
        windows_pileup = cid.pileup_patterns(all_pattern_windows)
        pileup_plot(windows_pileup, name=pileup_fname, output=output)


def cmd_test(args):

    sys.stderr.write(f"Fetching test dataset at {URL_EXAMPLE_DATASET}...\n")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    cio.download_file(URL_EXAMPLE_DATASET, tmp.name)

    sys.stderr.write(f"Running detection on test dataset...\n")

    args["<contact_map>"] = tmp.name
    args["--no-plotting"] = True
    cmd_detect(args)
    os.unlink(tmp.name)


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


def main():
    args = docopt.docopt(__doc__, version=__version__)
    detect = args["detect"]
    generate_config = args["generate-config"]
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
    elif quantify:
        cmd_quantify(args)
    return 0


if __name__ == "__main__":
    main()
