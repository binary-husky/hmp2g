import numpy as np
import timeit
"""
    calculate distance matrix for a position vector array A, support 3d and 2d
"""
def distance_matrix(X):
    """
    Calculates the pairwise distance matrix for a set of points in N-dimensional space.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The set of points to calculate the pairwise distances for.

    Returns:
    --------
    D : array, shape (n_samples, n_samples)
        The pairwise distance matrix.
    """
    n_samples = X.shape[0]
    XX = np.sum(X**2, axis=1).reshape(-1,1) 
    dis = np.sqrt(np.maximum(XX + XX.T - 2*X.dot(X.T),0)) # aij^2 + aji^2 - 2*aij*aji
    return dis
"""
[[0.         1.41421356 2.        ]
 [1.41421356 0.         1.41421356]
 [2.         1.41421356 0.        ]]
"""


"""
    calculate distance matrix for a position vector array A, support 3d and 2d
"""
def my_distance_matrix(A):
    n_subject = A.shape[-2]  # is 2
    A = np.repeat(np.expand_dims(A, -2), n_subject, axis=-2)  # =>(64, 100, 100, 2)
    At = np.swapaxes(A, -2, -3)  # =>(64, 100, 100, 2)
    dis = At - A  # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis


# B = np.array([  [0,-1],
#                 [1, 0],
#                 [0, 1],])

# print(distance_matrix(B))

# Generate some random data
X = np.random.rand(5, 3).astype(np.float32)
print(np.isclose(distance_matrix(X),my_distance_matrix(X)))
# # Measure the time it takes to calculate the distance matrix
# t = timeit.timeit(lambda: distance_matrix(X), number=10);print(t)
t = timeit.timeit(lambda: my_distance_matrix(X), number=10);print(t)
