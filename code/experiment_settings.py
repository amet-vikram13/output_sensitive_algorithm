#!/usr/bin/env python3
# -*- coding: utf-8 -*-

data_path = "/data/local/AA/data/"
results_path = "/data/local/AA/results/"

# specify list of sample sizes m
M = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

# number of repetitions per experiment
repetitions = 50

# renaming some data sets
data_name = { # old:new
    "ijcnn1": "Ijcnn1",
    "pose": "Pose",
    "song": "Song",
    "covertype": "Covertype",
}
