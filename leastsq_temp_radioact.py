import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

# 2.2
def least_temp_resistors(T, R, t=0):
    func_quad = lambda coefs, x : coefs[0] * x * x + coefs[1] * x + coefs[2]
    error = lambda coefs, x, y : func_quad(coefs, x) - y

    coefs_initial = (1.0, 1.0, 1.0)
    coefs_final, success = leastsq(error, coefs_initial[:], args=(T, R))
    print("Quadratic fit : ", coefs_final)

    x = np.linspace(T.min(), T.max(), 50)
    y = func_quad(coefs_final, x)

    f = plt.figure()
    plt.plot(x, y, 'r-', label='Approximated function$')
    plt.plot(T, R, 'bo', label='Real given points')
    plt.xlabel('T')
    plt.ylabel('R')
    plt.title("Least square Quadratic approximation")
    plt.legend(loc='lower left')
    plt.grid(True)
    f.savefig('least_temp_resistor.png')
    if t != 0:
        return max(error(coefs_final, x, y)), func_quad(coefs_final, t)
    else:
        return max(error(coefs_final, x, y))

def least_radioactif(T, N):
    func = lambda params, t : params[0] * np.exp(-params[1] * t)
    error = lambda params, x, y : func(params, x) - y

    params_initial = (0, 0)
    params_final, success = leastsq(error, params_initial[:], args=(T, N))
    print("Radioactive Law fit (parameter) : N(0) = ", params_final[0], 'lambda = ', params_final[1])

    x = np.linspace(T.min(), T.max(), 50)
    y = func(params_final, x)

    t_half = np.log(2) / params_final[1]

    f = plt.figure()
    plt.plot(x, y, 'r-', label='Approximated function$')
    plt.plot(T, N, 'bo', label='Real given points')
    plt.xlabel('T')
    plt.ylabel('N')
    plt.title("Least square Quadratic approximation")
    plt.legend(loc='lower left')
    plt.grid(True)
    f.savefig('least_radioactif.png')
    return params_final[1], t_half, max(error(params_final, x, y))


if __name__ == '__main__':
    
    # # Least square (Quadratic approximation) Temperature-Resistor
    T = np.array([20, 22, 24, 26, 28])
    R = np.array([40, 21.1, 11.1, 5.9, 3.1])
    r = least_temp_resistors(T, R, t=21)
    print('\nThe total quadratic residual of the resulting function')
    print('Residual =', r[0], 'and Temperature for t=21 is T =', r[1])
    print()

    # # Least square (Quadratic approximation) radioactif half-life of isotopes
    T = np.array([0, 0.913088, 1.826176, 2.739264, 3.652352, 4.56544])
    N = np.array([1000981, 943007, 892128, 840575, 795179, 748918])

    res = least_radioactif(T, N)
    print('\nThe parameter calculating of the function')
    print('Decay constant (lambda) = ', res[0])
    print('The half-life T(1/2) = ', res[1])
    print('Residual = ', res[2])