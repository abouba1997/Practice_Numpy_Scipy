import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as lin
from scipy.sparse import dia_matrix
import timeit

# 1.2

def odu4(coefs, func, L, phi, psi, N):
    a4, a3, a2, a1, a0 = coefs
    lphi, rphi = phi
    lpsi, rpsi = psi
    h = L / N

    A0 = np.ones(N + 1)  # y[i]
    Au1 = np.zeros(N + 1) # y[i+1]
    Au2 = np.zeros(N + 1) # y[i+2]
    Ad1 = np.zeros(N + 1) # y[i-1]
    Ad2 = np.zeros(N + 1) # y[i-2]

    A0[:] = 6 * a4 / h ** 4 - 2 * a2 / h ** 2 + a0
    Au1[:] = -4 * a4 / h ** 4 - a3 / h ** 3 + a2 / h ** 2 + a1 / (2 * h)
    Au2[:] = a4 / h ** 4 + a3 / (2 * h ** 3)
    Ad1[:] = -4 * a4 / h ** 4 + a3 / h ** 3 + a2 / h ** 2 - a1 / (2 * h)
    Ad2[:] = a4 / h ** 4 - a3 / (2 * h ** 3)
    F = np.fromfunction(func, (N + 1, 1))

    A0[0] = 1
    Au1[0] = 0
    Au2[0] = 0
    Ad1[0] = 0
    Ad2[0] = 0
    F[0] = lphi

    A0[1] = 1 / h
    Au1[1] = 0
    Au2[1] = 0
    Ad1[1] = -1 / h
    Ad2[1] = 0
    F[1] = lpsi

    A0[N] = 1
    Au1[N] = 0
    Au2[N] = 0
    Ad1[N] = 0
    Ad2[N] = 0
    F[N] = rphi

    A0[N - 1] = -1 / h
    Au1[N - 1] = 1 / h
    Au2[N - 1] = 0
    Ad1[N - 1] = 0
    Ad2[N - 1] = 0
    F[N - 1] = rpsi

    Au1 = np.roll(Au1, 1)
    Au2 = np.roll(Au2, 2)
    Ad1 = np.roll(Ad1, -1)
    Ad2 = np.roll(Ad2, -2)
    A_band = np.concatenate((Au2, Au1, A0, Ad1, Ad2)).reshape(5, N + 1)

    res = lin.solve_banded((2, 2), A_band, F)
    return res

def residual(coefs, L, phi, psi, N):
    res = []
    for n in N:
        r = np.array(odu4(coefs, lambda x, y: -x * L / n, L, phi, psi, n)).flatten()
        res.append(r)

    residues = []
    for i in range(len(res) - 1):
        r1 = np.array(res[i]).flatten()
        r2 = np.array(res[i+1][::len(res[0]) - 1]).flatten()
        residues.append(max(np.abs(r1 - r2)))

    return residues

def analyse_odu4(N, L, filename):
    # u'''' + u' + u = x
    t1 = timeit.default_timer()
    y1 = odu4((1, 0, 0, 1, 1), lambda x, y: x * L / N, L, (0, 0), (0, 0), N)
    t1_final = timeit.default_timer() - t1
    
    f1 = plt.figure()
    plt.plot(np.linspace(0, L, N + 1), y1, 'b-')
    f1.savefig(filename)
    print("\nTime taking by program for N = ", str(N))
    print("Method solve_banded: t = ", t1_final)


def solution_matlab(filename): # just a test
    y = []
    with open(filename) as f:
        for i in f:
            y.append(float(i))
    return y


if __name__ == '__main__':
    N = [100, 1000, 10000]
    L = 1

    # Time and calulation for differents N
    for n in N:
        analyse_odu4(n, L, 'graph_odu4_' + str(n) + '.png')

    # Residues calculation
    resi = residual((1, 0, 0, 1, 1), L, (0, 0), (0, 0), [10, 100, 1000, 10000])
    print("\nResidual for N = " + ", ".join([str(i) for i in N]))
    print(*resi, sep='\t')

    p = -(np.log10(resi[1]) - np.log10(resi[0])) / (np.log10(100) - np.log10(10))
    print("\nApproximation order")
    print("p = ", np.round(p))


    # Comparaison with matlab solution in file odu4.txt with N = 100
    filename = 'F:/MATLAB/Code/odu4.txt'
    y = solution_matlab(filename)
    y1 = odu4((1, 0, 0, 1, 1), lambda x, y: x * L / 100, L, (0, 0), (0, 0), 100)
    f1 = plt.figure()
    plt.plot(np.linspace(0, L, 101), y1, 'b-', label='program solution')
    plt.plot(np.linspace(0, L, 101), y, 'r--', label='Matlab solution')
    plt.legend(loc='lower left')
    f1.savefig('solution_matlab_python.png')
