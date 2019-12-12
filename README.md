# Chromosight
<img src="docs/chromosight.gif" alt="animated logo" width="200"/>

[![PyPI version](https://badge.fury.io/py/chromosight.svg)](https://badge.fury.io/py/chromosight) [![Build Status](https://travis-ci.com/koszullab/chromosight.svg?branch=master)](https://travis-ci.com/koszullab/chromosight) [![codecov](https://codecov.io/gh/koszullab/chromosight/branch/master/graph/badge.svg)](https://codecov.io/gh/koszullab/chromosight) [![Read the docs](https://readthedocs.org/projects/chromosight/badge)](https://chromosight.readthedocs.io) [![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 

Detect chromatin loops (and other patterns) in Hi-C contact maps.

## Installation

```sh
pip3 install -U chromosight
```

or, if you want to get the very latest version:

```
sudo pip3 install -e git+https://github.com/koszullab/chromosight.git@master#egg=chromosight
```

## Usage

`chromosight` has 3 subcommands: `detect`, `quantify` and `generate-config`. To get the list and description of thos subcommands, you can always run:

```bash
chromosight --help
```
Detailed help for each subcommand can be displayed by running e.g. `chromosight detect --help`. Pattern detection is done using the `detect` subcommand.

```
chromosight detect <contact_maps> [<output>] [--kernels=None] [--loops]
                       [--borders] [--precision=4] [--iterations=auto]
                       [--output]
```

## Input

Input Hi-C contact maps can be either in bedgraph2d or cool format. Bedgraph2d is defined as a tab-separated text file with 7 columns: chr1 start1 end1 chr2 start2 end2 contacts. The cool format is an efficient and compact format for Hi-C data based on HDF5. It is maintained by the Mirny lab and documented here: https://mirnylab.github.io/cooler/

## Output
Two files are generated in the output directory (replace pattern by the pattern used, e.g. loops or borders):
  * `pattern_out.txt`: List of genomic coordinates, bin ids and correlation scores for the pattern identified
  * `pattern_out.json`: JSON file containing the windows (of the same size as the kernel used) around the patterns from pattern.txt

Alternatively, one can set the `--win-fmt=npy` option to dump windows into a npy file instead of JSO. This format can easily be loaded into a 3D array using numpy's `np.load` function.

## Options

```
Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect <contact_map> [<output>] [--kernel-config=FILE]
                        [--pattern=loops] [--precision=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--subsample=no] [--inter]
                        [--min-dist=0] [--max-dist=auto] [--no-plotting] [--dump=DIR]
                        [--min-separation=auto] [--threads=1] [--n-mads=5]
                        [--resize-kernel] [--perc-undetected=auto]
    chromosight generate-config <prefix> [--preset loops]
    chromosight quantify [--pattern=loops] [--inter] [--subsample=no] [--n-mads=5]
                         [--win-size=auto] <bed2d> <contact_map> <output>

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

```

### Contributing

All contributions are welcome. We use the [numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings when documenting functions.

The code formatting standard we use is [black](https://github.com/psf/black), with --line-length=79 to follow PEP8 recommendations. We use `nose2` as our testing framework. Ideally, new functions should have associated unit tests, placed in the `tests` folder.

To test the code, you can run:

```bash
nose2 -s tests/
```
