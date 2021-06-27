import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import eigvals

class LotkaVolterra:
    def __init__(self, a, b, c, d, e):
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e
        self.X_f0 = np.array([0., 0.])
        self.X_f1 = np.array([self.c / (self.d * self.b), self.a / self.b])
        self.t = np.linspace(0, 15, 1000)   # time
        self.X0 = np.array([10, 5])              # initials conditions (10 rabbits, 5 foxes)
    
    def dX_dt(self, X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([self.a * X[0] - self.b * X[0] * X[1] ,
                        -self.c * X[1] + self.d * self.b * X[0] * X[1] ])

    def modified_dX_dt(self, X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([self.a * X[0] - self.b * X[0] * X[1] - self.e * X[0] ** 2,
                        -self.c * X[1] + self.d * self.b * X[0] * X[1] ])


    def d2X_dt2(self, X, t=0):
        """ Return the Jacobian matrix evaluated in X. """
        return np.array([[self.a - self.b * X[1],   -self.b * X[0]],
                  [self.b * self.d * X[1],   -self.c + self.b * self.d * X[0]] ])

    def analysis(self):
        # Equilibrium
        print("\nEquilibrium status: ", end=' ')
        print(all(system.dX_dt(self.X_f0) == np.zeros(2)) and all(system.dX_dt(self.X_f1) == np.zeros(2)), end=' ')
        print("for ", self.X_f0, "and for ", self.X_f1)

        # Stability fixed points
        A_f0 = system.d2X_dt2(self.X_f0)
        A_f1 = system.d2X_dt2(self.X_f1)
        print("\nStability fixed points")
        print(A_f0)
        print()
        print(A_f1)
        print()

        # Periodicity of the population
        lambda1, lambda2 = eigvals(A_f1)
        print("The eigenvalues are: ", lambda1, lambda2, "(Imaginaries numbers)")
        print("The fox and rabbit populations are periodic as follows from further")
        T_f1 = 2 * np.pi / abs(lambda1)
        print("Their period is given by: T = ", T_f1)

        X, infodict = odeint(self.dX_dt, self.X0, self.t, full_output=True)
        return X

    def plot_populas(self, filename='rabbits_and_foxes_populas.png'):
        X = self.analysis()
        if X is None:
            return None

        rabbits, foxes = X.T
        f = plt.figure()
        plt.plot(self.t, rabbits, 'r-', label='Rabbits')
        plt.plot(self.t, foxes  , 'b-', label='Foxes')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('population')
        plt.title('Evolution of fox and rabbit populations')
        f.savefig(filename)


    def plot_fields_and_trajectory(self, filename='rabbits_and_foxes_fields.png'):
        values  = np.linspace(0.3, 0.9, 5)                            # position of X0 between X_f0 and X_f1
        vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))  # colors for each trajectory
        f = plt.figure()
        #-------------------------------------------------------
        # plot trajectories
        for v, col in zip(values, vcolors):
            X0 = v * self.X_f1                                 # starting point
            X = odeint(self.dX_dt, self.X0, self.t)            # we don't need infodict here
            plt.plot(X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.f, %.f)' % (X0[0], X0[1]))

        #-------------------------------------------------------
        # define a grid and compute direction at each point
        ymax = plt.ylim(ymin=0)[1]                           # get axis limits
        xmax = plt.xlim(xmin=0)[1]
        nb_points = 20

        x = np.linspace(0, xmax, nb_points)
        y = np.linspace(0, ymax, nb_points)
    
        X1 , Y1  = np.meshgrid(x, y)                         # create a grid
        DX1, DY1 = self.dX_dt([X1, Y1])                      # compute growth rate on the gridt
        M = (np.hypot(DX1, DY1))                             # Norm of the growth rate 
        M[ M == 0] = 1.                                      # Avoid zero division errors 
        DX1 /= M                                             # Normalize each arrows
        DY1 /= M

        #-------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        # I choose to plot normalized arrows and to use colors to give information on
        # the growth speed
        plt.title('Trajectories and direction fields')
        Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=plt.cm.jet)
        plt.xlabel('Number of rabbits')
        plt.ylabel('Number of foxes')
        plt.legend()
        plt.grid()
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        f.savefig(filename)

    def IF(self, X):
        u, v = X
        return u ** (self.c / self.a) * v * np.exp( -(self.b / self.a) * (self.d * u + v))
    
    def plot_contour(self, filename='rabbits_and_foxes_contour.png'):
        nb_points = 80                                    # grid size
        ymax = plt.ylim(ymin=0)[1]                        # get axis limits
        xmax = plt.xlim(xmin=0)[1]
        x = np.linspace(0, xmax, nb_points)
        y = np.linspace(0, ymax, nb_points)
        X2 , Y2  = np.meshgrid(x, y)                     # create the grid
        Z2 = self.IF([X2, Y2])                           # compute IF on each point
        f = plt.figure()
        CS = plt.contourf(X2, Y2, Z2, cmap=plt.cm.Purples_r, alpha=0.5)
        CS2 = plt.contour(X2, Y2, Z2, colors='black', linewidths=2. )
        plt.clabel(CS2, inline=1, fontsize=16, fmt='%.f')
        plt.grid()
        plt.xlabel('Number of rabbits')
        plt.ylabel('Number of foxes')
        plt.ylim(1, ymax)
        plt.xlim(1, xmax)
        plt.title('IF contours')
        f.savefig(filename)

if __name__ == '__main__':
    a, b, c, d = 1, 0.1, 1.5, 0.75
    system = LotkaVolterra(a, b, c, d)
    system.plot_populas()
    system.plot_fields_and_trajectory()
    system.plot_contour()
    
