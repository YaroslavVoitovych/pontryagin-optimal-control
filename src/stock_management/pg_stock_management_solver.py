import numpy as np

from dataclasses import dataclass
from typing import Callable


from src.ode_utils import solve_ivp


@dataclass
class StockManagementProblem:
    T: np.float16
    a: np.float16
    x_initial: np.float16
    u_1: np.float16
    u_2: np.float16
    u_initial: np.float16
    c_1: np.float16
    c_2: np.float16
    g_1: np.float16 = None
    g_2: np.float16 = None
    g_3: np.float16 = None
    g_4: np.float16 = None
    h_state: np.float16 = 1e-3
    h_gradient: np.float16 = 1e-3
    eps_cost_derivative: np.float16 = 1e-3
    ro_init: np.float16 = 1
    b: np.float16 = 0.6
    eps_steplength: np.float16 = 1e-3
    max_ro: np.int16 = 20
    max_iter: np.int16 = 100

    def __post_init__(self):
        self.g_t = lambda t: self.g_1*t + self.g_2 if self.T/2 >= t >= 0 else self.g_3 - self.g_4 * t
        self.psi_x = lambda x: self.c_1 * x ** 2 if x >= 0 else self.c_2 * x ** 2
        self.psi_x_derivative = lambda x: 2 * self.c_1 * x if x >= 0 else 2 * self.c_2 * x


def norm_cost_derivative(cost_derivative_u):
    return sum([x**2 for x in cost_derivative_u])


def norm_gradient_stop_condition(norm_gradient,  eps_cost_derivative):
    return norm_cost_derivative(norm_gradient) < eps_cost_derivative


def gradient_descent_loop_stop_condition(cost_derivative_u, eps_cost_derivative, current_rho, eps_steplength,
                                         current_iter, max_iter):
    stop_condition = (norm_gradient_stop_condition(cost_derivative_u, eps_cost_derivative) & (current_rho < eps_steplength) &
                      (current_iter < max_iter))
    return stop_condition


def gradient_projection_operator(u1, u2):
    def projection_operator_with_params(omega):
        if u1 <= omega <= u2:
            return omega
        elif omega < u1:
            return u1
        else:
            return u2
    return projection_operator_with_params


def gradient_descent_loop(x_initial, u_initial, ro_init, a, u_1, u_2, T, g_t, psi_x, psi_x_derivative, max_rho, b_rho,
                            h_state, h_gradient, eps_cost_derivative, eps_steplength, max_iter):

    # Присвоєння початкових умов
    j = 0
    cost_derivative_u = [np.inf, ]
    new_cost = cost = np.inf
    current_rho = ro_init
    u_j_new = u_j = u_initial

    # На кожній ітерації перевірка умов збіжності алгоритму
    while not gradient_descent_loop_stop_condition(cost_derivative_u, eps_cost_derivative, current_rho,
                                                   eps_steplength, j, max_iter):
        print(f'Main loop. Cost: {new_cost}')
        print(f'Main loop. Rho: {current_rho}')
        # Розв'язок рівняння стану
        x_j = solve_ivp(lambda t, state: u_j - g_t(t), x_initial, T, h_state)
        print(x_j) # ??
        # Розв'язок рівняння спряженого стану
        p_j = solve_ivp(lambda t, state: -psi_x_derivative(x_j[int(t/h_state)]), 0.0, T, h_state, backward=True)
        print(p_j)
        # Обчислення градієнита функції втрат з урахуванням аналітичної формули похідної по керуванню
        cost_derivative_u = p_j + a

        # Перевірка умови на норму градієнта
        if norm_gradient_stop_condition(cost_derivative_u, eps_cost_derivative):
            break

        # Пошук кроку градієнтного спуску
        # На кожній ітерації виконується перевірка на величину поточного кроку та на приріст функції втрат
        gradient_projection_function = gradient_projection_operator(u_1, u_2)
        for i in range(0, max_rho):
            print(f'Rho loop. Cost: {new_cost}')
            print(f'Rho loop. Rho: {current_rho}')
            if current_rho < eps_steplength:
                break

            # Обчислення нового керування
            u_j_new = gradient_projection_function(u_j - current_rho * cost_derivative_u)
            # Розв'язок нового рівняння стану
            x_j_new = solve_ivp(lambda t, state: u_j_new - g_t(t), x_initial, T, h_state)

            # Обчислення нового значення функції втрат
            t_range = np.arange(0, T, h_state)
            cost_func = lambda t: a * u_j_new + psi_x(x_j_new[int(t/h_state)])
            new_cost = np.trapz(y=cost_func(t_range), x=t_range, dx=h_state)
            if new_cost >= cost:
                current_rho *= b_rho
            else:
                break
    return u_j_new, new_cost


if __name__ == '__main__':
    sm_problem_params = StockManagementProblem(T=12.0, a=0.1, x_initial=5, u_1=10.0, u_2=16.0, u_initial=14.0,
                                               c_1=1e-3, c_2=3e-3, g_1=1.0, g_2=10.0, g_3=22.0, g_4=1.0)

    u, cost = gradient_descent_loop(sm_problem_params.x_initial, sm_problem_params.u_initial, sm_problem_params.ro_init,
                          sm_problem_params.T, sm_problem_params.a, sm_problem_params.u_1, sm_problem_params.u_2,
                          sm_problem_params.g_t, sm_problem_params.psi_x, sm_problem_params.psi_x_derivative,
                          sm_problem_params.max_ro, sm_problem_params.b,
                          sm_problem_params.h_state, sm_problem_params.h_gradient,
                          sm_problem_params.eps_cost_derivative, sm_problem_params.eps_steplength,
                          sm_problem_params.max_iter, )
