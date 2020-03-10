# Chromosight
<img src="docs/chromosight.gif" alt="animated logo" width="200"/>

[![PyPI version](https://badge.fury.io/py/chromosight.svg)](https://badge.fury.io/py/chromosight) [![Anaconda cloud](https://anaconda.org/bioconda/chromosight/badges/version.svg)](https://anaconda.org/bioconda/chromosight) [![Build Status](https://travis-ci.com/koszullab/chromosight.svg?branch=master)](https://travis-ci.com/koszullab/chromosight) [![codecov](https://codecov.io/gh/koszullab/chromosight/branch/master/graph/badge.svg)](https://codecov.io/gh/koszullab/chromosight) [![Read the docs](https://readthedocs.org/projects/chromosight/badge)](https://chromosight.readthedocs.io) [![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 

Python package to detect chromatin loops (and other patterns) in Hi-C contact maps.

## Installation

Stable version with pip:

```sh
pip3 install --user chromosight
```
Stable version with conda:
```sh
conda install -c bioconda -c conda-forge chromosight
```

or, if you want to get the latest development version:

```
pip3 install --user -e git+https://github.com/koszullab/chromosight.git@master#egg=chromosight
```

## Usage

`chromosight` has 3 subcommands: `detect`, `quantify` and `generate-config`. To get the list and description of those subcommands, you can always run:

```bash
chromosight --help
```
Pattern detection is done using the `detect` subcommand. The generate-config subcommand is used to create a new type of pattern that can then be fed to `detect` using the `--custom-kernel` option. The `quantify` subcommand is used to compute pattern matching scores for a list of 2D coordinates on a Hi-C matrix.

### Get started
To get a first look at a chromosight run, you can run `chromosight test`, which will download a test dataset from the github repository and run `chromosight detect` on it.

### Important options

* `--min-dist`: Minimum distance from which to detect patterns.
* `--max-dist`: Maximum distance from which to detect patterns. Increasing also increases runtime and memory use.
* `--pearson`: Decrease to allow a greater number of pattern detected (with potentially more false positives).
* `--perc-undetected`: Proportion of empty pixels allowed in a window for detection.

### Example

To detect all chromosome loops with sizes between 2kb and 200kb using 8 parallel threads:
```bash
chromosight detect --threads 8 --min-dist 20000 --max-dist 200000 hic_data.cool out_dir
```

## Input

Input Hi-C contact maps should be in cool format. The cool format is an efficient and compact format for Hi-C data based on HDF5. It is maintained by the Mirny lab and documented here: https://mirnylab.github.io/cooler/

Most other Hi-C data formats (hic, homer, hic-pro), can be converted to cool using [hicexplorer's hicConvertFormat](https://hicexplorer.readthedocs.io/en/latest/content/tools/hicConvertFormat.html). Bedgraph2 format can be converted directly using cooler with the command `cooler load -f bg2 <chrom.sizes>:<binsize> in.bg2.gz out.cool`. For more informations, see the [cooler documentation](https://cooler.readthedocs.io/en/latest/cli.html#cooler-load)

## Output
Two files are generated in the output directory (replace pattern by the pattern used, e.g. loops or borders):
  * `pattern_out.txt`: List of genomic coordinates, bin ids and correlation scores for the pattern identified
  * `pattern_out.json`: JSON file containing the windows (of the same size as the kernel used) around the patterns from pattern.txt

Alternatively, one can set the `--win-fmt=npy` option to dump windows into a npy file instead of JSON. This format can easily be loaded into a 3D array using numpy's `np.load` function.

## Options

```
Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops] [--pearson=auto]
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

```

### Contributing

All contributions are welcome. We use the [numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings when documenting functions.

The code formatting standard we use is [black](https://github.com/psf/black), with --line-length=79 to follow PEP8 recommendations. We use `nose2` as our testing framework. Ideally, new functions should have associated unit tests, placed in the `tests` folder.

To test the code, you can run:

```bash
nose2 -s tests/
```
