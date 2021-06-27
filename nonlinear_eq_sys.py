import numpy as np
from scipy.optimize import fsolve

# 2.1

def func(x):
    return 2 * np.exp(x) + 3 * x + 1

def func_2variable(x):
    return [
        np.sin(x[1] + 1) - x[0] - 1.2,
        2 * x[1] + np.cos(x[0]) - 2
    ]

def root(func, x_approx):
    r = fsolve(func, x_approx)
    return r


if __name__ == '__main__':
    print("\nFor one variable function (2*exp(x) + 3*x + 1 = 0)")
    print('Root = ', root(func, -0.5), end=' ')
    print("\n\nFor two variable function (sin(y + 1) - x = 1.2, 2*y + cos(x) = 2)")
    print('Roots = ', root(func_2variable, [-0.5, 0.5]), end=' ')
    print()