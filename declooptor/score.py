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
    list_predicted = list(map(tuple, list_predicted))
    list_real = list(map(tuple, list_real))
    set_predicted = set(list_predicted)
    set_real = set(list_real)
    nb_loops_found = 0
    for pred in set_predicted:
        is_real = False    
        for real in set_real:
            if (int(pred[0]) in range(int(real[0])-area,int(real[0])+area+1)) and (int(pred[1]) in range(int(real[1])-area,int(real[1])+area+1)):
                nb_loops_found +=1
                is_real = True
                set_real.remove(real)
            if is_real:
                break
 
    if len(list_predicted) > 0 :      
        PREC = float(nb_loops_found) / len(list_predicted)  # consider that each pixel predicted will correspond to a different loop 
    else :
        PREC = "NA"    
    if  len(list_real) > 0 :
        RECALL = float(nb_loops_found) / len(list_real)
    else : 
        RECALL = "NA"
    if  PREC != "NA" and RECALL != "NA"  and PREC != 0 and RECALL != 0 :  
        F1 =     2* (PREC * RECALL) / (PREC + RECALL)
    else : 
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
