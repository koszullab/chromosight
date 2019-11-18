
![Chromosight logo](docs/chromosight.gif)

# Chromosight


[![PyPI version](https://badge.fury.io/py/chromosight.svg)](https://badge.fury.io/py/chromosight)
[![Build Status](https://travis-ci.com/koszullab/chromosight.svg?branch=master)](https://travis-ci.org/koszullab/chromosight)
[![Read the docs](https://readthedocs.org/projects/chromosight/badge)](https://chromosight.readthedocs.io)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Detect chromatin loops (and other patterns) in Hi-C contact maps.

## Installation

```sh
    pip3 install -U chromosight
```

or, if you want to get the very latest version:

```sh
   sudo pip3 install -e git+https://github.com/koszullab/chromosight.git@master#egg=chromosight
```

## Usage

    chromosight detect <contact_maps> [<output>] [--kernels=None] [--loops]
                           [--borders] [--precision=4] [--iterations=auto]
                           [--output]

## Input

Input Hi-C contact maps can be either in bedgraph2d or cool format. Bedgraph2d is defined as a tab-separated text file with 7 columns: chr1 start1 end1 chr2 start2 end2 contacts. The cool format is an efficient and compact format for Hi-C data based on HDF5. It is maintained by the Mirny lab and documented here: https://mirnylab.github.io/cooler/

## Output
Two files are generated in the output directory (replace pattern by the pattern used, e.g. loops or borders):
  * pattern.txt: List of genomic coordinates, bin ids and correlation scores for the pattern identified
  * pattern.json: JSON file containing the windows (of the same size as the kernel used) around the patterns from pattern.txt

## Options

```
Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect <contact_map> [<output>] [--kernel-config FILE]
                        [--pattern=loops] [--precision=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--subsample=no]
                        [--inter] [--max-dist=auto] [--no-plotting] [--threads 1]
    chromosight generate-config <prefix> [--preset loops]

    detect: 
        performs pattern detection on a Hi-C contact map using kernel convolution
    generate-config:
        Generate pre-filled config files to use for `chromosight detect`. 
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
    -n, --no-plotting           Disable generation of pileup plots.
    -P, --pattern loops         Which pattern to detect. This will use preset
                                configurations for the given pattern. Possible
                                values are: loops, borders, hairpin. [default: loops]
    -p, --precision auto        Precision threshold when assessing pattern
                                probability in the contact map. A lesser value
                                leads to potentially more detections, but more
                                false positives. [default: auto]
    -s, --subsample=INT         Subsample contacts from the matrix to INT contacts.
                                This is useful when comparing matrices with different
                                coverages. [default: no]
    -t, --threads 1             Number of CPUs to use in parallel. [default: 1]
    -w, --win-fmt={json,npy}    File format used to store individual windows
                                around each pattern. Window order match
                                patterns inside the associated text file.
                                Possible formats are json and npy. [default: json]

Arguments for generate-config:
    prefix                      Path prefix for config files. If prefix is a/b,
                                files a/b.json and a/b.1.txt will be generated.
                                If a given pattern has N kernel matrices, N txt
                                files are created they will be named a/b.[1-N].txt.
    -p, --preset loops          Generate a preset config for the given pattern.
                                Preset configs available are "loops" and 
                                "borders". [default: loops]
                                
```
