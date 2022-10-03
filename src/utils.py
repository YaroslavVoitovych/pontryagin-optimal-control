import os
import imageio
import numpy as np
import matplotlib.pyplot as plt


def l2_norm(vector):
    if not isinstance(vector, np.ndarray):
        return vector
    else:
        return np.sum(np.array([np.power(x, 2) for x in vector]))


def laplacian_operator_approximation_2d(state_t, h):
    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    transformed_state = np.zeros_like(state_t)

    for x_1 in range(transformed_state.shape[0]):
        for x_2 in range(transformed_state.shape[1]):
            transformed_state[x_1, x_2] = ((state_t[x_1 - 1, x_2] if x_1 != 0 else state_t[x_1, x_2]) +
                                           (state_t[x_1 + 1, x_2] if x_1 != state_t.shape[0]-1 else state_t[x_1, x_2]) +
                                           (state_t[x_1, x_2 - 1] if x_2 != 0 else state_t[x_1, x_2]) +
                                           (state_t[x_1, x_2 + 1] if x_2 != state_t.shape[1]-1 else state_t[x_1, x_2]) -
                                           4 * state_t[x_1, x_2])/(h ** 2)

    return transformed_state


def laplacian_operator_approximation_1d(state_t, h):
    transformed_state = np.zeros_like(state_t)
    for x_1 in range(transformed_state.shape[0]):
        if x_1 == 0:
            transformed_state[x_1] = (state_t[x_1] + state_t[x_1 + 1] - 2 * state_t[x_1]) / (h ** 2)
        elif x_1 == transformed_state.shape[0] - 1:
            transformed_state[x_1] = (state_t[x_1 - 1] + state_t[x_1] - 2 * state_t[x_1]) / (h ** 2)
        else:
             transformed_state[x_1] = (state_t[x_1 - 1] + state_t[x_1 + 1] - 2 * state_t[x_1]) / (h ** 2)
    return transformed_state


def viz_1d_control(time: np.array, control: np.array):
    plt.title("Оптимальне керування u(t)")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.plot(time, control, c='maroon', linewidth=3)
    plt.grid()
    plt.show()


def viz_2d_heatmap(matrix):
    fig, ax = plt.subplots()
    plot = ax.imshow(matrix)
    ax.invert_yaxis()
    plt.colorbar(plot, ax=ax)
    plt.show()
    return fig



def viz_2d_time_gif(image_list, model_name='None'):

    # build gif
    with imageio.get_writer(f'/Users/yvoitovych/Desktop/repos/pontryagin-optimal-control/plots/{model_name}.gif', mode='I') as writer:
        for filename in image_list:
            image = imageio.v2.imread(f'/Users/yvoitovych/Desktop/repos/pontryagin-optimal-control/plots/{filename}')
            writer.append_data(image)

    # Remove files
    for filename in set(image_list):
        os.remove(filename)


def save_slice_plot(matrix):
    viz_2d_heatmap(matrix)
    plt.savefig('')