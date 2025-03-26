from clarkson_coreset_algorithm.clarkson import *
from utils import *
import matplotlib.pyplot as plt


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
    assert isConvexCombination(X, ind_E, s) is None
    print("----- isConvexCombination() passed -----\n")

# For array X = [[1,2,3],[3,4,5],[5,6,7]]
# array [5,6,7] is not a convex combination
# of remaining two.
def test_isNotConvexCombination():
    X = getPreData_X()
    ind_E = [0, 1]
    s = 2
    assert isConvexCombination(X, ind_E, s) is not None
    print("----- isNotConvexCombination() passed -----\n")

def test_isConvexCombination_data(path):
    X_c = np.load(path).get("X")
    non_convex_pts = []
    for i in range(len(X_c)):
        print("Processing",i,"th point")
        ind_E = np.arange(len(X_c)).tolist()
        ind_E.remove(i)
        if isConvexCombination(X_c, ind_E, i) is not None :
            non_convex_pts.append(i)
            print(non_convex_pts)
    np.savez(
        coresets_path + "_clarkson_cs_non_convex_pts.npz",
        X=non_convex_pts
    )
    print("----- isConvexCombination_data() complete -----\n")

def test_ijcnn1_convex_combination_upper_bound():
    X, y = load_data("ijcnn1")

    s = np.random.choice(X.shape[0])

    ind_E = np.setdiff1d(np.arange(len(X)), s).tolist()

    t_start = time()
    print("Is point s convex combination of remaining points:",isConvexCombination(X, ind_E, s) is None)
    t_end = time()

    print("Time taken: ", t_end-t_start)

    print("----- ijcnn1_convex_combination_upper_bound() passed -----\n")

def test_song_convex_combination_upper_bound():
    X, y = load_data("song")

    s = np.random.choice(X.shape[0])

    ind_E = np.setdiff1d(np.arange(len(X)), s).tolist()

    t_start = time()
    print("Is point s convex combination of remaining points:",isConvexCombination(X, ind_E, s) is None)
    t_end = time()

    print("Time taken: ", t_end-t_start)

    print("----- ijcnn1_convex_combination_upper_bound() passed -----\n")

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
    X_C = clarksonCoreset(X, ind_E, ind_S, "ijcnn1", "CK")
    t_end = time()

    print("Length of X_C: ", len(X_C))
    print("Time taken: ", t_end-t_start)

    print("----- ijcnn1_clarkson_coreset() passed -----\n")

def test_ijcnn1_clarkson_coreset_using_TA_CK(m=1000):
    X, y = load_data("ijcnn1")

    X = X[np.random.choice(X.shape[0], m, replace=False)]

    print("Applying clarkson coreset algorithm using CK")
    # takes too long to run
    t_start = time()
    X_C_1 = computeClarksonCoreset(X, "ijcnn1", "CK")
    t_end = time()

    print("Applying clarkson coreset algorithm using TA")
    # takes too long to run
    t_start = time()
    X_C_2 = computeClarksonCoreset(X, "ijcnn1", "TA")
    t_end = time()

    print("Length of CK X_C: ", len(X_C_1))
    print("Time taken: ", t_end-t_start)

    print("Length of TA X_C: ", len(X_C_2))
    print("Time taken: ", t_end - t_start)

    print("----- TA CK algo check passed -----\n")

def test_unitBall(n,dim=2,method="CK"):
    X = np.random.normal(0, 1, (n, dim))
    X = np.apply_along_axis(lambda v:  v / np.linalg.norm(v, ord=2), 1, X)
    radii = np.random.random(n) ** (1/dim)
    X = X * radii.reshape(-1, 1)

    X_C = computeClarksonCoreset(X, None, method)

    plt.figure(figsize=(10, 10))
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', linewidth=2)
    plt.gca().add_patch(circle)

    # Plot the original points
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.3, color='blue', label='Original Points')

    # Plot the coreset points (size proportional to weight)
    plt.scatter(X_C[:, 0], X_C[:, 1], s=10, color='red', label=f'Coreset Points ({len(X_C)} points)')

    # Plot settings
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(f'Clarkson Coreset with {method} method')
    plt.legend()
    plt.axis('equal')

    # Save the plot
    plt.savefig(f"unit_ball_test_{method}.png")
    plt.close()

    print("----- test_unitball passed -----\n")

def run_tests():
    test_farthestPointsSetUsingMinMax()
    test_isConvexCombination()
    test_isNotConvexCombination()

    ## Special cases
    # test_ijcnn1_convex_combination_upper_bound()
    # test_song_convex_combination_upper_bound()
    # test_ijcnn1_clarkson_coreset(m=1679)
    # test_isConvexCombination_data(results_path + "ijcnn1_clarkson_coreset.npz")
    # test_ijcnn1_clarkson_coreset_using_TA_CK(2000)
    # test_unitBall(1000,dim=2,method="TA")


if __name__ == "__main__":
    run_tests()
