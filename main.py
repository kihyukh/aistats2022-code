import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def dimensionality_measure(A, T, gamma):
    n, p = A.shape
    X = cp.Variable((p, p), symmetric=True)
    P = cp.Variable(n)
    P_diag = cp.diag(P)
    S = A.T @ (P_diag * T / gamma) @ A + np.eye(p)
    prob = cp.Problem(
        cp.Minimize(cp.trace(X)),
        [
            P >= 0,
            cp.sum(P) <= 1,
            cp.bmat([
                [X, np.eye(p)],
                [np.eye(p), S],
            ]) >> 0,
        ]
    )
    prob.solve()
    return n - prob.value

def maximum_information_gain(A, T, gamma):
    n, p = A.shape
    P = cp.Variable(n)
    P_diag = cp.diag(P)
    S = A.T @ (P_diag * T / gamma) @ A + np.eye(p)
    prob = cp.Problem(
        cp.Maximize(cp.log_det(S)),
        [
            P >= 0,
            cp.sum(P) <= 1,
        ]
    )
    prob.solve()
    return prob.value / 2


if __name__ == '__main__':
    np.random.seed(2)
    n = 50
    d = 5
    T = 1000
    gamma = 1
    A = np.random.randn(n, d)
    A /= np.sqrt(np.square(A).sum(axis=1))[:, np.newaxis]
    K = pairwise_kernels(A, metric='rbf', gamma=0.1)
    w, v = np.linalg.eigh(K)
    Phi = np.zeros((n, n))
    for i in range(n):
        a = v[:, i]
        Phi += np.sqrt(max(0, w[i])) * np.outer(a, a)
    print(dimensionality_measure(Phi, T, gamma))
    print(maximum_information_gain(Phi, T, gamma))