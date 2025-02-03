#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from time import time

from utils import *


def compute_clarkson_coreset(dataset):
    X, y = load_data(dataset)  # y won't be used

    t_start = time()

    X_C = []

    t_end = time()

    print("Length of X_C: ", len(X_C))
    print("Time taken: ", t_end-t_start)

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    compute_clarkson_coreset(dataset)
