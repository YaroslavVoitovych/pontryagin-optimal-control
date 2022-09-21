import numpy as np
import matplotlib.pyplot as plt


def l2_norm(vector):
    return sum([x ** 2 for x in vector])


def viz_1d_control(time: np.array, control: np.array):
    plt.title("Оптимальне керування u(t)")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.plot(time, control, c='maroon', linewidth=3)
    plt.grid()
    plt.show()
