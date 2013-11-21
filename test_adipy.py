""" test_AD.py: test AD functionality with toy problem """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from adipy import *
import adipy.linalg
from time import time

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    # control points
    N = 32

    # solution tolerance
    tol = 1.0e-8

    # create x vector and d/dx^2 operator
    x, D, I = chebyshev(N, True)
    D2 = np.dot(D,D)

    # initial guess
    f0 = np.zeros(N - 2)

    # call solver
    solution = root(residuals, f0, args=D2, method="hybr", tol=tol, jac=J)
    
    # append BCs
    f = np.append(0.0, solution.x)
    f = np.append(f, 0.0)

    # plot results     
    plt.plot(x, f, 'o-')
    title = r"$\frac{d^2 f}{d x^2} = e^{\pi f}$"
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title, fontsize=18)
    plt.grid(True)       
    plt.show()
    
    return

def residuals(f, D2):
    # parameter
    alpha = np.pi

    # append BCs
    f = np.append(0.0, f)
    f = np.append(f, 0.0)

    # compute residuals
    R = np.dot(D2, f) - exp(alpha*f)

    return R[1:-1]

def J(f, D2):
    fi = ad(f, np.eye(len(f)))
    res = jacobian(residuals(fi, D2))
    return res

def chebyshev(N, integration=False):
    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    # initialize
    D = np.zeros((N, N))

    # x array
    x = 0.5*(1 - np.cos(np.pi*np.arange(0, N)/(N - 1)))

    # D operator
    c = np.array(2)
    c = np.append(c, np.ones(N - 2))
    c = np.append(c, 2)
    c = c*((-1)**np.arange(0, N))
    A = np.tile(x, (N, 1)).transpose()
    dA = A - A.transpose() + np.eye(N)
    cinv = 1/c

    for i in range(N):
        for j in range(N):
            D[i, j] = c[i]*cinv[j]/dA[i, j]

    D = D - np.diag(np.sum(D.transpose(),axis=0))

    # I operator
    if integration:
        I = adipy.linalg.inv(D[1:, 1:])
        I = np.append(np.zeros((1, N - 1)), I, axis=0)
        I = np.append(np.zeros((N, 1)), I, axis=1)
        return x, D, I
    else:
        return x, D

# call main
if __name__ == '__main__':
    main()
