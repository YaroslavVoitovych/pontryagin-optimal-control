import os
import imageio
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm

PLOTS_DIR = '/Users/yvoitovych/Desktop/repos/pontryagin-optimal-control/plots/'



def l2_norm(vector):
    if not isinstance(vector, np.ndarray):
        return vector
    else:
        return np.sum(np.array([np.power(x, 2) for x in vector]))


def laplacian_operator_approximation_2d(state_t, h):
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


def viz_1d_control(time: np.array, control: np.array, title='None', x_label='None', y_label='None'):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(time, control, c='red', linewidth=3)
    plt.grid()
    plt.show()


def viz_2d_heatmap(matrix, name, save=False):
    if name is None:
        name = 'UNDEFINED'
    fig, ax = plt.subplots()
    plot = ax.imshow(matrix)
    ax.invert_yaxis()
    plt.colorbar(plot, ax=ax)
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, f'heatmap_{name}_{datetime.datetime.now().timestamp()}.png'))
        plt.show()
    return fig


def viz_2d_time_gif(image_list: list, model_name='None'):
    with imageio.get_writer(os.path.join(PLOTS_DIR, f'{model_name}_{datetime.datetime.now().timestamp()}.gif'), mode='I') as writer:
        for idx, matrix in enumerate(image_list):
            fig = viz_2d_heatmap(matrix, str(idx))
            plt.savefig(os.path.join(f'{PLOTS_DIR}', f'{idx}.png'))
            image = imageio.v2.imread(os.path.join(f'{PLOTS_DIR}', f'{idx}.png'))
            writer.append_data(image)
    for idx, _ in enumerate(image_list):
        os.remove(os.path.join(f'{PLOTS_DIR}', f'{idx}.png'))


def viz_3d_plot(matrix, name, save=False):
    x = range(matrix.shape[0])
    y = range(matrix.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    surf = ax.plot_surface(X, Y, matrix,rstride=1, cstride=1, cmap=cm.turbo, linewidth=0, antialiased=False)
    ax.set_xlabel('Просторова координата x1, см')
    ax.set_ylabel('Просторова координата x2, см')
    ax.set_zlabel('Значення функції стану системи')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(os.path.join(PLOTS_DIR, f'3D_{name}_{datetime.datetime.now().timestamp()}.png'), )
    plt.show()


def viz_1d_compare(time: np.array, matrix1: np.array, matrix2: np.array, title1='None', title2='None', x_label='None', y_label='None'):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(time, matrix1, c='maroon', linewidth=3, label=title1)
    plt.scatter(time[::5], matrix2[::5], c='blue', linewidth=3, label=title2)
    plt.legend()
    plt.grid()
    plt.show()
