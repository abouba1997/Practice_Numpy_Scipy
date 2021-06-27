import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SIR:
    def __init__(self, beta, gamma, S0, I0, R0):
        self.beta, self.gamma, self.S0, self.I0, self.R0 = beta, gamma, S0, I0, R0

    def odes(self, X, t):
        S, I, R = X[0], X[1], X[2]

        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
    
        return [dSdt, dIdt, dRdt]

    def solve_odes(self, t):
        X0 = [self.S0, self.I0, self.R0]
        X = odeint(self.odes, X0, t)
        return X
    
    def analysis_sir(self, filename='sir_model.png'):
        t = np.linspace(0, 60, 100)
        res = np.array(self.solve_odes(t))
        S, I, R = res[:, 0], res[:, 1], res[:, 2]

        # result plotting
        f = plt.figure()
        plt.plot(t, S, label='Subjected')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Removed')
        plt.legend()
        f.savefig(filename)


if __name__ == '__main__':
    beta, gamma = 0.0005, 0.1
    S0, I0, R0 = 1500, 1, 0

    system = SIR(beta, gamma, S0, I0, R0)
    system.analysis_sir()
    