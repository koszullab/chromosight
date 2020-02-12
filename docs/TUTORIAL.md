# Tutorial

The simplest way to run chromosight without any input data is to use:

```bash
chromosight test
```

Which will download a test dataset and run chromosight on it. This is useful to have a look at the output files.

`chromosight detect` takes input in the form of Hi-C matrices, either in cool or bedgraph2 format. This command allows to detect patterns on Hi-C maps, such as chromatin loops or domain (e.g. TADS) borders, and report their coordinates.

The following command line can be used to run loop detection (the default pattern):

```bash
chromosight detect -t12 some_sample.cool results
```

The program will run in parallel on 12 threads and write loop coordinates in a file named `loops_out.txt` inside the `results` folder. Similarly, to run domain borders detection, one can use:

```bash
chromosight detect --pattern borders -t12 some_sample.cool results
```

Which will write the coordinates of borders in `results/borders_out.txt`.

At this point, the `results` folder will also contain files `loops_out.json` and `borders_out.json`, which contain images of the matrix region around each detected loop or border, respectively. These files are in JSON format, which can be natively loaded in most programming languages.

chromosight has many command line options which can affect the output format or filter the output based on different criteria. All parameters have default parameter values defined for each pattern, but these can be overriden by the user to optimize results if needed.
