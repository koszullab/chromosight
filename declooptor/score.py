#!/usr/bin/env python3
# coding: utf-8

"""Pattern scoring

Validate predicted patterns and simulated patterns at pixel scale through
a variety of metrics (false positive rate, false negative rate, etc.) for
benchmarking purposes.

    Usage:
        score.py <predicted_patterns> <real_patterns> [--area=3] [--size=1000]


    Arguments:
        predicted_patterns              A text file containing in two
                                        columns the predicted pattern
                                        coordinates in the matrix.                       
        real_patterns                   A file like predicted_patterns, except
                                        with the actual pattern coordinates.

    Options:
        -h, --help                      Display this help message.
        --version                       Display the program's current version.
        -a 3, --area 3                  Pattern area (overlap will determine
                                        a match). [default: 3]
        -s, --size 1000                 Initial matrix size. [default: 1000]

"""

import numpy as np
import docopt
from declooptor.version import __version__


def score_loop(list_predicted, list_real, n1, area):
    MAT_PREDICT = np.zeros(
        (n1 + area * 2, n1 + area * 2)
    )  # load into memory all the predicted loops in one matrix
    for l in list_predicted:
        x = int(l[0])
        y = int(l[1])
        MAT_PREDICT[
            np.ix_(
                range(x - area, x + area + 1), range(y - area, y + area + 1)
            )
        ] += 1

    nb_loops_found = 0
    for l in list_real:
        x = int(l[0])
        y = int(l[1])
        bool_find = 0
        MAT_REAL = np.zeros((n1 + area * 2, n1 + area * 2))
        MAT_REAL[
            np.ix_(
                range(x - area, x + area + 1), range(y - area, y + area + 1)
            )
        ] += 1
        bool_find = (MAT_REAL * MAT_PREDICT).sum()
        if bool_find > 0:
            nb_loops_found += 1

    if len(list_predicted) > 0:
        PREC = nb_loops_found / float(
            len(list_predicted)
        )  # consider that each pixel predicted will match a different loop
    else:
        PREC = "NA"
    if len(list_real) > 0:
        RECALL = nb_loops_found / float(len(list_real))
    else:
        RECALL = "NA"
    if PREC != "NA" and RECALL != "NA" and PREC != 0 and RECALL != 0:
        F1 = 2 * (PREC * RECALL) / (PREC + RECALL)
    else:
        F1 = "NA"

    return PREC, RECALL, F1


def main():
    arguments = docopt.docopt(__doc__, version=__version__)
    predicted_patterns = arguments["<predicted_patterns>"]
    real_patterns = arguments["<real_patterns>"]
    area = arguments["--area"]
    size = arguments["--size"]

    score_loop(predicted_patterns, real_patterns, area, size)


if __name__ == "__main__":
    main()
