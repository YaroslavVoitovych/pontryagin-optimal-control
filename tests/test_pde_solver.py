import sys

import numpy as np
import matplotlib.pyplot as plt
from src.utils import laplacian_operator_approximation_1d, laplacian_operator_approximation_2d, viz_2d_time_gif, viz_2d_heatmap
from src.ode_utils import solve_ivp

t_eps = 0.1
s_eps = 0.01
k = 1000
r = 1.0
gamma = 0.0006


if __name__ == '__main__':
    L = 1
    T = 20
    N = int(L/s_eps)
    R_omega = 0.4
    x_1_omega = 0.5
    x_2_omega = 0.5


    def gaussian(x1, x2):
        return 1 * np.exp((-(x1 - x_1_omega) ** 2)/(2*0.25*R_omega**2)) * np.exp((-(x2 - x_2_omega) ** 2)/(2*0.01*R_omega**2))




    #y_initial = np.repeat(np.array([(1 - i/2)**2 for i in np.linspace(0, 10, N)]), N, axis=0).reshape((N, N))
    y_initial = np.zeros(shape=(N, N))
    for x1 in range(y_initial.shape[0]):
        for x2 in range(y_initial.shape[1]):

            y_initial[x1, x2] = 10 * (y_initial.shape[1] - x2)/y_initial.shape[1] #gaussian(x1*s_eps, x2*s_eps) #if (x1 * eps - x_1_omega) ** 2 + \
                                              # (x2 * eps - x_2_omega) ** 2 < R_omega ** 2 else 0

    u = -1000 * np.ones(shape=(int(T/t_eps), *y_initial.shape))
    subspace_mask = np.zeros(shape=y_initial.shape)
    for x1 in range(subspace_mask.shape[0]):
        for x2 in range(subspace_mask.shape[1]):
            subspace_mask[x1, x2] = 1 if (int(x1 * s_eps) - x_1_omega) ** 2 + \
                                      (int(x2 *s_eps) - x_2_omega) ** 2 < R_omega ** 2 \
        else 0

    def state_equation(t, state):
        return r * state * (1 - state / k) + gamma * laplacian_operator_approximation_2d(state, s_eps) - \
               subspace_mask * np.sin(t) * u[int(t / t_eps), :, :] * \
               state


    viz_2d_heatmap(y_initial)
    plt.show()

    #sys.exit()
    print(y_initial.shape)
    # y_initial *= np.ones(shape=(int(L/eps)))

    state_projection = solve_ivp(state_equation, y_initial, T, t_eps)
    print(state_projection.shape)
    TN = int(T/t_eps)
    viz_2d_time_gif([f'{i}.png' for i in range(0, TN-1)], 'test')
    #plt.plot(state_projection[:, 0])
    #fig, ax = plt.subplots()

    #ax.imshow(state_projection, cmap='RdYlGn')
    #ax.invert_yaxis()
    #plt.colorbar()
    #plt.show()
