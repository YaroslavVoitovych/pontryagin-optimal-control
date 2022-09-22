import imageio
import numpy as np
import matplotlib.pyplot as plt


def l2_norm(vector):
    return sum([x ** 2 for x in vector])


def laplacian_operator_approximation_2d(state_t, h):
    #def wrapper(x_1, x_2):
    transformed_state = np.zeros_like(state_t)
    print(transformed_state.shape)
    for x_1 in range(transformed_state.shape[0]):
        for x_2 in range(transformed_state.shape[1]):
            transformed_state[x_1, x_2] = (state_t[x_1 - 1, x_2] if x_1 - 1 >= 0 else 0 +
                                state_t[x_1 + 1, x_2] if x_1 + 1 < state_t.shape[0] else 0 +
                                state_t[x_1, x_2 - 1] if x_2 - 1 >= 0 else 0 +
                                state_t[x_1, x_2 + 1] if x_1 + 1 < state_t.shape[1] else 0 -
                                4 * state_t(x_1, x_2) if x_1 >= 0 and x_2 >= 0 else 0
                                                           )/(h ** 2)
    return transformed_state
    #return wrapper


def viz_1d_control(time: np.array, control: np.array):
    plt.title("Оптимальне керування u(t)")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.plot(time, control, c='maroon', linewidth=3)
    plt.grid()
    plt.show()


def viz_2d_heatmap(matrix):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.show()


def viz_2d_time_gif(matrix):
    pass
