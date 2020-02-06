# -*- coding: utf-8 -*-
"""
Pattern scoring

Validate predicted patterns and simulated patterns through a variety of
metrics (false positive rate, false negative rate, etc.) for benchmarking
purposes.

    Usage:
        score_patterns.py <predicted_patterns> <target_patterns> 
                          [--area=3] [--size=1000] [--list]


    Arguments:
        predicted_patterns              A file containing two tab-separated
                                        columns with the predicted pattern
                                        coordinates in the matrix. If --list
                                        is enabled, this should be the path
                                        to a list of files, with one file per
                                        line.
        target_patterns                 Path to pattern file or list of files 
                                        like predicted_patterns, except
                                        with the actual pattern coordinates.

    Options:
        -h, --help                      Display this help message.
        -l, --list                      Input are list of pattern files instead
                                        of pattern files.
        -a 3, --area 3                  Pattern area (overlap will determine
                                        a match). [default: 3]
        -s, --size 289                  Initial matrix size. [default: 1000]

"""

import os
import sys
import re
import numpy as np
import pandas as pd
import time
import itertools
import pathlib
import warnings
import docopt

def load_benchmark_file(path, shape):
    """
    Perform sanity check to load and filter a pattern file.

    Parameters
    ----------
    path : str
        Path to the input file
    shape : tuple of ints
        Expected size of the matrix (i.e. highest coordinates allowed).
        Should be (m, n) where m and n are the number or rows and cols.

    Returns
    -------
    coords : numpy.array of ints
        2D array containing with 1 row per pattern. Columns 0 and 1
        contain horizontal and vertical coordinates, respectively.
        
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = np.loadtxt(path)
    # If nothing was detected, return an empty array
    if not len(coords):
        return np.array([[]])
    # sanity check on values
    else:
        # If only a single coordinate is present, reshape array to 2d
        if len(coords.shape) == 1:
            coords = coords.reshape((1, 2))
        # Same operation for x and y coordinates
        for ax in [0, 1]:
            # Remove spurious values coords outside matrix
            if np.any(coords[:, ax] < 0):
                coords = coords[coords[:, ax] >= 0, :]
            if np.any(coords[:, ax] >= shape[ax]):
                coords = coords[coords[:, ax] < SHAPE[ax], :]
        # Remove duplicate coordinates
        if coords.shape[0] > 1:
            if not (coords.shape[0] == np.unique(coords, axis=0).shape[0]):
                coords = np.unique(coords, axis=0)
    return coords


def fill_confusion_matrix(list_predicted, list_real, area):
    """
    Given a list of predicted pattern coordinates and real coordinates,
    compute the confusion matrix of loop events. Predicted coordinated
    within 'area' pixels of the real pattern are considered correct.

    Parameters
    ----------
    list_predicted : numpy.array of ints
    list_real : numpy.array of ints
    area : int
        Margin of error accepted around a real pattern for a predicted
        to be accepted.
    
    Returns
    -------
    conf : dict
        The confusion matrix, taking the form of a dictionary with
        keys TP, TN, FP and FN and the number of patterns falling in
        each category as values.
    """
    conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    list_predicted = list(map(tuple, list_predicted))
    list_real = list(map(tuple, list_real))
    set_predicted = set(list_predicted)
    set_real = set(list_real)
    # Compare each predicted pattern with each real pattern
    if len(list_predicted[0]):
        for pred in set_predicted:
            is_real = False
            for real in set_real:
                # Check if the predicted pattern is in an area around the real one
                if (
                    int(pred[0])
                    in range(int(real[0]) - area, int(real[0]) + area + 1)
                ) and (
                    int(pred[1])
                    in range(int(real[1]) - area, int(real[1]) + area + 1)
                ):
                    # A loop was found close to a target -> True Positive
                    conf["TP"] += 1
                    is_real = True
                    # Do not count two independent predictions correct for the
                    # same target
                    set_real.remove(real)
                if is_real:
                    break
            # A loop was predicted, but did not fall close to any
            # target -> False Positive
            if not is_real:
                conf["FP"] += 1
    # All targets that have not been found (and removed from the set) are False
    # Negatives
    conf["FN"] = len(set_real)
    return conf


def compute_metrics(conf):
    """
    Given a confusion matrix of loop events, compute precision,
    recall and F1 score.
    Parameters
    ----------
    conf : dict
        Confusion matrix in the form of a dictionary with keys
        TP, FP, TN, FN and integers as values.

    Returns
    -------
        prec : float or np.nan
            The proportion of predicted patterns that are real.
        recall : float or np.nan
            The proportion of real pattern that were detected.
        F1 : float or np.nan
            F1 score, a metric summarising precision and recall,
            defined as 2 * (prec * recall) / (prec + recall)
    """
    # Precision is only defined if some loops are detected
    if conf["TP"] + conf["FP"] > 0:
        # Precision: Proportion of predictions that are correct
        prec = conf["TP"] / (conf["TP"] + conf["FP"])
    else:
        prec = np.nan
    # Recall is only defined if there are targets
    if conf["TP"] + conf["FN"] > 0:
        # Recall: Proportion of targets that have been found
        recall = conf["TP"] / (conf["TP"] + conf["FN"])
    else:
        recall = np.nan
    if prec != np.nan and recall != np.nan and prec != 0 and recall != 0:
        F1 = 2 * (prec * recall) / (prec + recall)
    else:
        F1 = np.nan

    return prec, recall, F1


def main():
    args = docopt.docopt(__doc__)
    # Get pattern files from a list
    if args['--list']:
        predicted_fnames = open(args['<predicted_patterns>'], 'r').read()
        target_fnames = open(args['<target_patterns>'], 'r').read()
        # Exclude potential empty lines
        predicted_fnames = [f for f in predicted_fnames.split('\n') if f]
        target_fnames = [f for f in target_fnames.split('\n') if f]
    else:
        predicted_fnames = args["<predicted_patterns>"]
        target_fnames = args["<target_patterns>"]
    area = int(args["--area"])
    size = int(args["--size"])
    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for targ_f, pred_f in zip(target_fnames, predicted_fnames):
        # read target and prediction data files
        target = load_benchmark_file(targ_f, shape=(size, size))
        predict = load_benchmark_file(pred_f, shape=(size, size))
        # Compute confusion matrix for this run
        tmp_conf = fill_confusion_matrix(predict, target, area=area)
        # Increment global confusion matrix with the results
        for cat in confusion.keys():
            # TP, TN, FP, FN
            confusion[cat] += tmp_conf[cat]

    # Compute precision, recall and F1 score based on the confusion matrix built
    # on all Hi-C matrices
    precision, recall, f1 = compute_metrics(confusion)

    print("precision\trecall\tf1")
    print(f"{precision}\t{recall}\t{f1}")


if __name__ == "__main__":
    main()
