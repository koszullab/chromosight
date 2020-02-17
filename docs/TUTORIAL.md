# Tutorial

The simplest way to run chromosight without any input data is to use:

```bash
chromosight test
```
Which will download a test dataset and run chromosight on it. This is useful to have a look at the output files.

## Detection


`chromosight detect` takes input in the form of Hi-C matrices, either in cool or bedgraph2 format. This command allows to detect patterns on Hi-C maps, such as chromatin loops or domain (e.g. TADS) borders, and report their coordinates.

The following command line can be used to run loop detection (the default pattern):

```bash
chromosight detect -t12 sample1.cool results
```

The program will run in parallel on 12 threads and write loop coordinates and their pattern matching scores in a file named `loops_out.txt` inside the `results` folder. Those scores represent pearson correlation coefficients (i.e. between -1 and 1) between the loop kernel and the detected pattern.
Similarly, to run domain borders detection, one can use:

```bash
chromosight detect --pattern borders -t12 sample1.cool results
```

Which will write the coordinates and scores of borders in `results/borders_out.txt`.

At this point, the `results` folder will also contain files `loops_out.json` and `borders_out.json`, which contain images of the matrix regions around each detected loop or border, respectively. These files are in JSON format, which can be natively loaded in most programming languages.

Chromosight has many command line options which can affect the output format or filter the results based on different criteria. All parameters have sane default values defined for each pattern, which are printed during the run, but these can be overriden using command line options to optimize results if needed. The list of command line options can be shown using:

```bash
chromosight --help
```

## Quantification

The `chromosight quantify` command can be used to assign a pattern matching score to a set of 2D coordinates for an input Hi-C matrix. It will accept coordinates in bed2d format (tab-separated text file with 6 columns without headers, where columns denote chrom1, start1, end1, chrom2, start2, end2), or the output coordinates file `chromosight detect`. This can be useful to score the same set of coordinates on multiple Hi-C libraries, for instance.

For example, to compute loop scores for the positions detected in sample1.cool for a second sample, one could use:

```bash
chromosight quantify results/loops_out.txt sample2.cool results
```

Similarly, for borders:

```bash
chromosight quantify --pattern=borders results/borders_out.txt sample2.cool results
```

These commands will each generate two files in the `results` directory, named `loops_quant.txt` and `loops_quant.json` for the first command, and `borders_quant.txt` and `borders_quant.json` for the second. Those files have the same format as the output from `chromosight detect`.

`chromosight quantify` can also be useful to compute pattern scores at ChIP-seq peaks, genes, or other features of interest.

## Generating custom patterns

More advanced users with specific questions or problems may wish to create new patterns and configurations. Both `detect` and `quantify` will accept custom patterns through the `--kernel-config` option. In order to provide a custom pattern, the user needs 2 files:

* A JSON file containing default values for the different detection parameters.
* One or more text files containing the pattern kernel(s) (i.e. matrix) in the form of a dense numeric matrix.

A template configuration can be generated using `chromosight generate-config`. A preset on which the template will be based can be selected, `loops` being the default preset. For example, to generate a template config based on the borders pattern, the folowing command can be used:

```bash
chromosight generate-config --preset borders demo_pattern
```

This will generate a JSON file named `demo_pattern.json`, pre-filled with parameter values from the `borders` pattern. This JSON file will have the following contents:

```json
{
    "name": "borders",
    "kernels": [
        "demo_pattern.1.txt",
        "demo_pattern.2.txt",
        "demo_pattern.3.txt"
    ],
    "max_dist": 1,
    "min_dist": 0,
    "max_iterations": 3,
    "max_perc_undetected": 30.0,
    "min_separation": 5000,
    "precision": 0.3,
    "resolution": 5000
}

```

The user can edit the configuration parameters in a text editor. Notably, the `kernels` entry points to 3 files, `demo_pattern.[1-3].txt`, which have also been created by `chromosight generate-config`. Those 3 paths are relative to the config, which means the kernel files have to be in the same folder as the JSON config.

When given a config with multiple kernels, chromosight detect will scan the matrix once for each kernel and return the union of all detected coordinates for the different kernels. This is useful when a pattern is asymetric and can be flipped in different orientations, for example.

Kernels matrices are text files and can be edited using external program, or alternatively, the user can use the `--click` option from `generate-config` in order to manually build the kernel by double-clicking on relevant regions in a Hi-C matrix.

> Note: The `--click` option will consume lots of RAM as it visualises the entire Hi-C matrix and should be reserved for small or subsetted contact maps.

For example:

```bash
chromosight generate-config --click sample1.cool --win-size 15 demo_manual
```

This command will generate a config based on the loops template (the default) and will display the contact map `sample1.cool`. Every time the user double-clicks on a pixel, a window of 15x15 pixels centered on that position is recorded. The operation can be repeated as many times as the user wishes, and when the window is closed, all windows are averaged, a slight gaussian blur is added to reduce the impact of random noise, and the resulting pileup is used as the kernel when writing the config files.
