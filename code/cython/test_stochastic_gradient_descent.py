import numpy as np
import time
from stochastic_gradient_descent import stochastic_gradient_descent_not_optimized

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

n_iter = n # 9.5s for 1000000 loops
eta = 0.00005
m = 1

beta = stochastic_gradient_descent_not_optimized(eta, n_iter, x, y, n, p, m)
print(coefs)
print("\n----------------")
print(beta)
print("\n----------------")
print(sum((coefs - beta)**2))