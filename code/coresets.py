#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from archetypalanalysis import findWitnessVector
from archetypalanalysis import isConvexCombination
from experiment_settings import data_path
from time import time
from tqdm import tqdm


# "uniform" in the paper
def uniform_sample(X, m):
    n = X.shape[0]
    ind = np.random.choice(n, m)
    X_C = X[ind]
    return X_C


# "lw-cs" in the paper; outlined in Algorithm 1
def lightweight_coreset(X, m):
    # Scalable k-means clustering via lightweight coresets
    # Bachem et al. (2018)
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = 0.5 * 1 / n + 0.5 * dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C


# Mahalanobis D^2-sampling
# needed by lucic_coreset()
def mahanalobis_d2_sampling(X, k):
    n = X.shape[0]
    i = np.random.choice(n, 1)
    B = X[i]
    for _ in range(k - 1):
        # d_A(x,y) = is \|x-y\|_2^2
        dist = np.array(list(map(lambda b: np.sum((X - b) ** 2, axis=1), B)))
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, np.arange(n)]
        p = dist / dist.sum()
        i = np.random.choice(n, 1, replace=False, p=p)[0]
        B = np.vstack((B, X[i]))
    return B


# "lucic-cs" in the paper
def lucic_coreset(X, m, k):
    # Strong Coresets for Hard and Soft Bregman Clustering with Applications to Exponential Family Mixtures
    # Lucic et al. (2016)
    n = X.shape[0]
    B = mahanalobis_d2_sampling(X, k)
    a = 16 * (np.log(k) + 2)
    # d_A(x,y) = is \|x-y\|_2^2
    dist = np.array(list(map(lambda b: np.sum((X - b) ** 2, axis=1), B)))
    closest_cluster_id = dist.argmin(0)
    dist = dist[closest_cluster_id, np.arange(n)]
    c = dist.mean()
    s = np.zeros(n)
    for i in range(n):
        Bi_cardinality = np.sum(closest_cluster_id == closest_cluster_id[i])
        tmp = dist[closest_cluster_id == closest_cluster_id[i]].sum()
        s[i] = (
            a * dist[i] / c
            + 2 * a * tmp / (Bi_cardinality * c)
            + 4 * n / Bi_cardinality
        )
    p = s / s.sum()
    ind = np.random.choice(n, m, p=p)
    X_C = X[ind]
    w_C = 1 / (m * p[ind])
    return X_C, w_C


# proposed coreset
# "abs-cs" in the paper; outlined in Algorithm 2
def coreset(X, m):
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C

# proposed coreset
# "clarkson-cs" in the paper "More output-sensitive geometric algorithms"
def clarkson_coreset(X, ind_E, ind_S, dataset_name):
    X_C = np.empty(1,1)
    try:
        data = np.load(data_path + dataset_name + "_clarkson_coreset.npz")
        X_C = data["X"]
    except FileNotFoundError:
        t_start = time()
        try:
            with tqdm(total=len(ind_S)) as pbar:
                while len(ind_S) > 0:
                    s = ind_S.pop(0)
                    if not isConvexCombination(X, ind_E, s):
                        witness_vector = findWitnessVector(X, ind_E, s)
                        if witness_vector is not None:
                            max_dot_product = np.dot(witness_vector, X[s])
                            p_prime = None
                            for p in ind_S:
                                dot_product = np.dot(witness_vector, X[p])
                                if dot_product > max_dot_product:
                                    max_dot_product = dot_product
                                    p_prime = p
                            if p_prime is not None:
                                ind_E.append(p_prime)
                                ind_S.append(s)
                                ind_S.remove(p_prime)
                            else:
                                ind_E.append(s)
                    pbar.update(1)
        except Exception as e:
            print(e)
        X_C = X[ind_E].copy()
        t_end = time()
        np.savez(
            data_path + dataset_name + "_clarkson_coreset.npz",
            X=X_C,
            cs_time=t_end - t_start
        )
    finally:
        return X_C
