import numpy as np
import random
cimport cython

#@cython.boundscheck(False)
#@cython.wraparound(False)

def stochastic_gradient_descent_not_optimized(double eta, int n_iter, double[:,:]x, double[:]y, int n, int p, int m):

    cdef double[:] beta_no_cython = np.array([0 for _ in range(p)], dtype=np.float64)
    cdef double[:] y_hat_no_cython = np.array([0 for _ in range(n)], dtype=np.float64)
    cdef double[:] gradient_no_cython = np.array([0 for _ in range(p)], dtype=np.float64)
    cdef double[:] decrease_eta = np.logspace(0,3,n_iter)
    k = np.random.choice(n, n, replace=False)
    x_temp = np.asarray(x)
    y_temp = np.asarray(y)
    cdef double[:,:] x_shuffled = x_temp[k]
    cdef double[:] y_shuffled = y_temp[k]

    cdef int j = 1

    while j < n_iter:
        eta_j = eta / decrease_eta[j]
        x_j = x_shuffled[((j-1)*m):((j-1)*m+m)]
        y_j = y_shuffled[((j-1)*m):((j-1)*m+m)]
        for i in range(m):
            y_hat_no_cython[i] = 0
            for k in range(p):
                y_hat_no_cython[i] += x_j[i,k] * beta_no_cython[k]
        for k in range(p):
            gradient_no_cython[k] = 0
            for i in range(m):
                gradient_no_cython[k] += x_j[i,k] * (y_j[i] - y_hat_no_cython[i])
            gradient_no_cython[k] = (-2.0)/m * gradient_no_cython[k]
            beta_no_cython[k] = beta_no_cython[k] - eta_j * gradient_no_cython[k]
        j += 1

    return np.asarray(beta_no_cython)