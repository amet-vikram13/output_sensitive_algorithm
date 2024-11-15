import numpy as np

from experiments import *
from utils import *


def getPreData_X():
    return np.array([[1,2,3],[3,4,5],[5,6,7]])

def getPreData_ind_E():
    return [0, 2]

def getPreData_s():
    return 1

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# the farthest points are [1,2,3] and [5,6,7]
# so the indices of these points are 0 and 2
def test_farthestPointsSetUsingMinMax():
    X = getPreData_X()
    ind_E = farthestPointsSetUsingMinMax(X)
    assert ind_E == getPreData_ind_E()
    print("----- farthestPointsSetUsingMinMax() passed -----\n")

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# the convex combination of [3,4,5] is a
# convex combination of [1,2,3] and [5,6,7]
# with values of lambda as [0.5, 0.5]
def test_isConvexCombination():
    X = getPreData_X()
    ind_E = getPreData_ind_E()
    s = getPreData_s()
    assert isConvexCombination(X, ind_E, s)==True
    print("----- isConvexCombination() passed -----\n")

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# array [5,6,7] is not a convex combination
# of remaining two.
def test_isNotConvexCombination():
    X = getPreData_X()
    ind_E = [0, 1]
    s = 2
    assert isConvexCombination(X, ind_E, s)==False
    print("----- isNotConvexCombination() passed -----\n")

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# array [5, 6, 7] is not a convex combination
# and so a witness vector can be found, which maximizes
# the dot product of the witness vector with [5, 6, 7]
def test_findWitnessVector():
    X = getPreData_X()
    ind_E = [0,1]
    s = 2
    witness_vector = findWitnessVector(X, ind_E, s)
    assert witness_vector is not None
    print("Witness Vector: ", witness_vector)
    print("----- findWitnessVector() passed -----\n")

def test_ijcnn1_convex_combination_upper_bound():
    X, y = load_data("ijcnn1")

    s = np.random.choice(X.shape[0])

    ind_E = np.setdiff1d(np.arange(len(X)), s).tolist()

    t_start = time()
    print("Is point s convex combination of remaining points:",isConvexCombination(X, ind_E, s))
    t_end = time()

    print("Time taken: ", t_end-t_start)


def test_ijcnn1_witness_vector_upper_bound():
    X, y = load_data("ijcnn1")

    s = np.random.choice(X.shape[0])

    ind_E = np.setdiff1d(np.arange(len(X)), s).tolist()

    t_start = time()
    print("Witness Vector for point s:",findWitnessVector(X, ind_E, s))
    t_end = time()

    print("Time taken: ", t_end-t_start)

def test_ijcnn1_clarkson_coreset(m=1000):
    X, y = load_data("ijcnn1")

    X = X[np.random.choice(X.shape[0], m, replace=False)]

    print("Applying farthestPointsSetUsingMinMax algorithm")
    ind_E = farthestPointsSetUsingMinMax(X)
    print("Length of ind_E: ", len(ind_E))
    print(ind_E)
    ind_S = np.setdiff1d(np.arange(len(X)), np.array(ind_E)).tolist()

    print("Applying clarkson coreset algorithm")
    # takes too long to run
    t_start = time()
    X_C = clarkson_coreset(X, ind_E, ind_S, "ijcnn1")
    t_end = time()

    print("Length of X_C: ", len(X_C))
    print("Time taken: ", t_end-t_start)

def run_tests():
    test_farthestPointsSetUsingMinMax()
    test_isConvexCombination()
    test_isNotConvexCombination()
    test_findWitnessVector()
    # test_ijcnn1_convex_combination_upper_bound()
    # test_ijcnn1_witness_vector_upper_bound()
    # test_ijcnn1_clarkson_coreset(m=100) # takes too long to run

if __name__ == "__main__":
    run_tests()
