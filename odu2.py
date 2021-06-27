import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as lin
from scipy.sparse import dia_matrix
import timeit

# 1.1

def odu2(coefs, func, L, bcl, bcr, N):
    a, b = coefs
    lalpha, lbeta, lgamma = bcl
    ralpha, rbeta, rgamma = bcr
    h = L / N

    A0 = np.ones(N + 1)  # y[i]
    Au1 = np.zeros(N + 1) # y[i+1]
    Ad1 = np.zeros(N + 1) # y[i-1]

    Au1[:] = a / (2 * h) + 1 / h ** 2
    A0[:] = b - 2 / h ** 2
    Ad1[:] = -a / (2 * h) + 1 / h ** 2
    F = np.fromfunction(func, (N + 1, 1))

    A0[0] = lbeta - lgamma / h
    Ad1[0] = 0
    Au1[0] = lgamma / h
    F[0] = lalpha

    A0[N] = rbeta + rgamma / h
    Au1[N] = 0
    Ad1[N] = -rgamma / h
    F[N] = ralpha

    Au1 = np.roll(Au1, 1)
    Ad1 = np.roll(Ad1, -1)
    A_band = np.concatenate((Au1, A0, Ad1)).reshape(3, N + 1)

    res = lin.solve_banded((1, 1), A_band, F)
    return res

def odu2_solve(coefs, func, L, bcl, bcr, N):
    a, b = coefs
    lalpha, lbeta, lgamma = bcl
    ralpha, rbeta, rgamma = bcr
    h = L / N

    A0 = np.ones(N + 1)  # y[i]
    Au1 = np.zeros(N + 1) # y[i+1]
    Ad1 = np.zeros(N + 1) # y[i-1]

    Au1[:] = a / (2 * h) + 1 / h ** 2
    A0[:] = b - 2 / h ** 2
    Ad1[:] = -a / (2 * h) + 1 / h ** 2
    F = np.fromfunction(func, (N + 1, 1))

    A0[0] = lbeta - lgamma / h
    Ad1[0] = 0
    Au1[0] = lgamma / h
    F[0] = lalpha

    A0[N] = rbeta + rgamma / h
    Au1[N] = 0
    Ad1[N] = -rgamma / h
    F[N] = ralpha

    data = np.array([np.roll(Ad1, -1), A0, np.roll(Au1, 1)])
    offsets = np.array([-1, 0, 1])
    M = dia_matrix((data, offsets), shape=(N+1, N+1)).toarray()

    res = lin.solve(M, F)
    return res

def residual(coefs, L, bcl, bcr, N, solve_mod=odu2):
    res = []
    for n in N:
        r = np.array(solve_mod(coefs, lambda x, y: -x * L / n, L, bcl, bcr, n)).flatten()
        res.append(r)

    residues = []
    for i in range(len(res) - 1):
        r1 = np.array(res[i]).flatten()
        r2 = np.array(res[i+1][::len(res[0]) - 1]).flatten()
        residues.append(max(np.abs(r1 - r2)))

    return residues

def analyse_ode2(N, L, filename):
    # u´´+ u = -x
    t1 = timeit.default_timer()
    y1 = odu2([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N) # solve_banded
    t1_final = timeit.default_timer() - t1

    t2 = timeit.default_timer()
    y2 = odu2_solve([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N) # solve
    t2_final = timeit.default_timer() - t2
    
    f1 = plt.figure()
    plt.plot(np.linspace(0, L, N + 1), y1, 'b-', label='solve_banded')
    plt.plot(np.linspace(0, L, N + 1), y2, 'r--', label='solve')
    Ne = 10
    plt.plot(np.arange(0, L + L / Ne, L / Ne),
             np.fromfunction(lambda x, y: -2 * np.sin(x * L / Ne) - x * L / Ne,
             (Ne + 1, 1)), 'g.', label='solution')
    plt.legend(loc='lower left')
    f1.savefig(filename)
    print("\nTime taking by program for N = ", str(N))
    print("Method solve_banded: t = ", t1_final)
    print("Method solve       : t = ", t2_final)


if __name__ == '__main__':
    N = [100, 1000, 10000]
    L = np.pi

    # Time and calulation for differents N
    for n in N:
        analyse_ode2(n, L, 'graph_odu2_' + str(n) + '.png')

    # Residues calculation
    resi = residual([0, 1], L, (0, 1, 0), (1, 0, 1), [10, 100, 1000, 10000], solve_mod=odu2)
    print("\nResidual for N = " + ", ".join([str(i) for i in N]))
    print(*resi, sep='\t')

    p = -(np.log10(resi[2]) - np.log10(resi[1])) / (np.log10(1000) - np.log10(100))
    print("\nApproximation order")
    print("p = ", np.round(p))
