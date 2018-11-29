# Declooptor

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/declooptor.svg)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Detect chromatin loops (and other patterns) in Hi-C contact maps.

## Installation

```sh
    pip3 install -U declooptor
```

or, if you want to get the very latest version:

```sh
   sudo pip3 install -e git+https://github.com/koszullab/declooptor.git@master#egg=declooptor
```

## Usage

    declooptor.py detect <contact_maps> [<output>] [--kernels=None] [--loops]
                         [--borders] [--precision=4] [--iterations=auto]
                         [--output]

## Options

    -h, --help                  Display this help message.
    --version                   Display the program's current version.
    contact_maps                The Hi-C contact maps to detect patterns on, in
                                CSV format. File names must be separated by a
                                colon.
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
    -i auto, --iterations auto  How many iterations to perform after the first
                                template-based pass. Auto means iterations are
                                performed until convergence. [default: auto]
    -o, --output                Output directory to write the detected pattern
                                coordinates, agglomerated plots and matrix
                                images into.