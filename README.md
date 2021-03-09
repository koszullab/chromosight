# Chromosight
<img src="docs/logo/chromosight.gif" alt="animated logo" width="200"/>

[![PyPI version](https://badge.fury.io/py/chromosight.svg)](https://badge.fury.io/py/chromosight) [![Anaconda cloud](https://anaconda.org/bioconda/chromosight/badges/version.svg)](https://anaconda.org/bioconda/chromosight) [![Build Status](https://travis-ci.com/koszullab/chromosight.svg?branch=master)](https://travis-ci.com/koszullab/chromosight) [![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/koszullab/chromosight)](https://hub.docker.com/r/koszullab/chromosight) [![codecov](https://codecov.io/gh/koszullab/chromosight/branch/master/graph/badge.svg)](https://codecov.io/gh/koszullab/chromosight) [![Read the docs](https://readthedocs.org/projects/chromosight/badge)](https://chromosight.readthedocs.io) [![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/koszullab/chromosight.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/koszullab/chromosight/context:python)

Python package to detect chromatin loops (and other patterns) in Hi-C contact maps. 

* Associated publication: https://www.nature.com/articles/s41467-020-19562-7
* Documentation and analyses examples: https://chromosight.readthedocs.io
* scripts used for the analysis presented in the article https://github.com/koszullab/chromosight_analyses_scripts

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

The two most commonly subcommands of `chromosight` are `detect` and `quantify`. For more advanced use, there are two additional subcomands: `generate-config` and `list-kernels`. To get the list and description of those subcommands, you can always run:

```bash
chromosight --help
```
Pattern detection is done using the `detect` subcommand. The `quantify` subcommand is used to compute pattern matching scores for a list of 2D coordinates on a Hi-C matrix. The `generate-config` subcommand is used to create a new type of pattern that can then be fed to `detect` using the `--custom-kernel` option. The `list-kernels` command is used to view informations about the available patterns.

### Get started
To get a first look at a chromosight run, you can run `chromosight test`, which will download a test dataset from the github repository and run `chromosight detect` on it. You can then have a look at the output files generated.

### Important options

When running `chromosight detect`, there are a handful parameters which are especially important:

* `--min-dist`: Minimum genomic distance from which to detect patterns. For loops, this means the smallest loop size accepted (i.e. distance between the two anchors).
* `--max-dist`: Maximum genomic distance from which to detect patterns. Increasing also increases runtime and memory use.
* `--pearson`: Detection threshold. Decrease to allow a greater number of pattern detected (with potentially more false positives). Setting a very low value may actually reduce the number of detected patterns. This is due to the algorithm which might merge neighbouring patterns.
* `--perc-zero`: Proportion of zero pixels allowed in a window for detection. If you have low coverage, increasing this value may improve results.

### Example

To detect all chromosome loops with sizes between 2kb and 200kb using 8 parallel threads:
```bash
chromosight detect --threads 8 --min-dist 20000 --max-dist 200000 hic_data.cool output_prefix
```

## Options

```

Pattern exploration and detection

Explore and detect patterns (loops, borders, centromeres, etc.) in Hi-C contact
maps with pattern matching.

Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops]
                        [--pearson=auto] [--win-size=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--norm={auto,raw,force}]
                        [--subsample=no] [--inter] [--tsvd] [--smooth-trend]
                        [--n-mads=5] [--min-dist=0] [--max-dist=auto]
                        [--no-plotting] [--min-separation=auto] [--dump=DIR]
                        [--threads=1] [--perc-zero=auto]
                        [--perc-undetected=auto] <contact_map> <prefix>
    chromosight generate-config [--preset loops] [--click contact_map]
                        [--norm={auto,raw,norm}] [--win-size=auto] [--n-mads=5]
                        [--threads=1] <prefix>
    chromosight quantify [--inter] [--pattern=loops] [--subsample=no]
                         [--win-fmt=json] [--kernel-config=FILE] [--norm={auto,raw,norm}]
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

```

## Input

Input Hi-C contact maps should be in cool format. The cool format is an efficient and compact format for Hi-C data based on HDF5. It is maintained by the Mirny lab and documented here: https://open2c.github.io/cooler/

Most other Hi-C data formats (hic, homer, hic-pro), can be converted to cool using [hicexplorer's hicConvertFormat](https://hicexplorer.readthedocs.io/en/latest/content/tools/hicConvertFormat.html) or [hic2cool](https://github.com/4dn-dcic/hic2cool). Bedgraph2 format can be converted directly using cooler with the command `cooler load -f bg2 <chrom.sizes>:<binsize> in.bg2.gz out.cool`. For more informations, see the [cooler documentation](https://cooler.readthedocs.io/en/latest/cli.html#cooler-load)

For `chromosight quantify`, the bed2d file is a text file with at least 6 tab-separated columns containing pairs of coordinates. The first 6 columns should be `chrom start end chrom start end` and have no header. Alternatively, the output text file generated by `chromosight detect` is also accepted. Instructions to generate a bed2d file from a bed file are given [in the documentation](https://chromosight.readthedocs.io/en/stable/TUTORIAL.html#quantification).

## Output
Three files are generated by chromosight's `detect` and `quantify` commands. Their filenames are determined by the value of the `<prefix>` argument:
  * `prefix.tsv`: List of genomic coordinates, bin ids and correlation scores for the pattern identified
  * `prefix.json`: JSON file containing the windows (of the same size as the kernel used) around the patterns from pattern.txt
  * `prefix.pdf`: Plot showing the pileup (average) window of all detected patterns. Plot generation can be disabled using the `--no-plotting` option.

Alternatively, one can set the `--win-fmt=npy` option to dump windows into a npy file instead of JSON. This format can easily be loaded into a 3D array using numpy's `np.load` function.


### Contributing

All contributions are welcome. We use the [numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings when documenting functions.

The code formatting standard we use is [black](https://github.com/psf/black), with --line-length=79 to follow PEP8 recommendations. We use `nose2` as our testing framework. Ideally, new functions should have associated unit tests, placed in the `tests` folder.

To test the code, you can run:

```bash
nose2 -s tests/
```

### FAQ

Questions from previous users are available in the [github issues](https://github.com/koszullab/chromosight/issues?q=label%3Aquestion). You can open a new issue for your question if it is not already covered.
### Citation
When using Chromosight in you research, please cite the pubication: https://www.nature.com/articles/s41467-020-19562-7
