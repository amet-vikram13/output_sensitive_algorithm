#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from coresets import clarkson_coreset
from experiment_settings import *
from archetypalanalysis import farthestPointsSetUsingMinMax
from time import time
from utils import *

def compute_clarkson_coreset(dataset):
    X, y = load_data(dataset)  # y won't be used

    t_start = time()

    # initialize two extreme points via farthestPointsSetUsingMinMax function
    # maintain the initialized indices as set E. Note: len(E) < len(X)
    # maintain the indices not belonging to E as set S. Note: len(S) = len(X) - len(E)
    # any index not belonging to E is a candidate for the next coreset
    ind_E = farthestPointsSetUsingMinMax(X)
    ind_S = np.setdiff1d(np.arange(len(X)), np.array(ind_E)).tolist()

    # obtain initial coreset using Clarkson's algorithm
    X_C = clarkson_coreset(X, ind_E, ind_S, dataset)

    t_end = time()

    print("Length of X_C: ", len(X_C))
    print("Time taken: ", t_end-t_start)

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    compute_clarkson_coreset(dataset)
