from archetypalanalysis import farthestPointsSetUsingMinMax
import numpy as np

def getPreData():
    return np.array([[1,2,3],[3,4,5],[5,6,7]])

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# the farthest points are [1,2,3] and [5,6,7]
# so the indices of these points are 0 and 2
def test_farthestPointsSetUsingMinMax():
    X = getPreData()
    ind_E = farthestPointsSetUsingMinMax(X)
    assert ind_E == [0, 2]
    print("----- farthestPointsSetUsingMinMax() passed -----")

def run_tests():
    test_farthestPointsSetUsingMinMax()

if __name__ == "__main__":
    run_tests()
