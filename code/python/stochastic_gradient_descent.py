import numpy as np
import time
import random

np.random.seed(0)

n = 50000000 # nombre de points
p = 8 # nombre de features

coefs = np.random.randn(p) * 3 # coeficients de la fonction f(x) = a0 + a1x1 + a2x2 + ... + apxp (loi normale N(0,3^2))

def f(x):
    return [np.dot(z,coefs) for z in x]

x = np.zeros((n,p))
x[:,0] = np.ones(n)
x[:,1:] = np.random.uniform(0,10,(n,p-1))

epsilon = np.random.randn(n) * 2 # Ajout du epsilon
y = f(x) + epsilon
y = np.array(y)

################## Descente de gradient stochastique #######################

def stochastic_gradient_descent_not_optimized(eta, n_iter, m):
    #k = random.sample(range(n), n)
    k = np.random.choice(n, n, replace=False)
    x_shuffled = x[k]
    y_shuffled = y[k]
    beta_no = [0 for _ in range(p)]
    y_hat_no = [0 for _ in range(n)]
    gradient_no = [0 for _ in range(p)]
    decrease_eta = np.logspace(0,3,n_iter)
    i = 1
    while i < n_iter:
        eta_i = eta / decrease_eta[i]
        x_i = x_shuffled[((i-1)*m):((i-1)*m+m)]
        y_i = y_shuffled[((i-1)*m):((i-1)*m+m)]
        for l in range(m):
            y_hat_no[l] = 0
            for k in range(p):
                y_hat_no[l] += x_i[l,k] * beta_no[k]
        for k in range(p):
            gradient_no[k] = 0
            for l in range(m):
                gradient_no[k] += x_i[l,k] * (y_i[l] - y_hat_no[l])
            gradient_no[k] = gradient_no[k] * (-2/m)
            beta_no[k] = beta_no[k] - eta_i * gradient_no[k]
        i += 1   
    return beta_no

n_iter = n # 27s for 1000000 loops
eta = 0.00005
blocs = 1

beta = stochastic_gradient_descent_not_optimized(eta, n_iter, blocs)
print(coefs)
print("\n----------------")
print(beta)
print("\n----------------")
print(sum((coefs - beta)**2))
