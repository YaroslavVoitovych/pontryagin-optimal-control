import logging
import numpy as np
import matplotlib.pyplot as plt


from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.pg_pmp_solver import PMPPDESolver
from src.utils import laplacian_operator_approximation_1d


@dataclass
class OptimalHarvestingLogisticProblem1D:
    # Термінальні значення для часу і простору
    T: np.float16
    B: np.float16
    y_initial: np.float16
    u_initial: np.float16
    # Параметри рівняння Фішера
    gamma: np.float16
    r: np.float16
    k: np.float16
    # Визначимо підпростір омега, як круг радіуса R_omega з центром в x_1_omega, x_2_omega:
    x_1_omega: np.float16
    x_2_omega: np.float16
    # Межі функції керування
    u_1: np.float16
    u_2: np.float16
    # Кроки дискретизації по часу та простору
    h_state_time: np.float16 = 1e-2
    h_state_space: np.float16 = 1e-2
    # Технічні параметри для алгоритму Ерміджо
    eps_cost_derivative: np.float16 = 1e-3
    ro_init: np.float16 = 1
    b: np.float16 = 0.6
    eps_steplength: np.float16 = 1e-3
    max_ro: np.int16 = 20
    max_iter: np.int16 = 100

    def __post_init__(self):
        self.y_initial *= np.ones(shape=(int(self.B/self.h_state_space),))
        self.u_initial *= np.ones(shape=(int(self.T/self.h_state_time), *self.y_initial.shape))
        # створюємо бінарну маску ядрової функції m, що задає підпростір омега, на якому застосовується керування
        self.subspace_mask = np.zeros(shape=self.y_initial.shape)
        for x1 in range(self.subspace_mask.shape[0]):
            self.subspace_mask[x1] = 1 if self.x_1_omega * self.subspace_mask.shape[0] < \
                                          x1 < self.x_2_omega * self.subspace_mask.shape[0] else 0
        self.u_initial *= self.subspace_mask

        # Початковий розподіл популяції по ареалу
        self.space_function = lambda x_1: self.y_initial[int(x_1/self.h_state_space)]


if __name__ == '__main__':
    ohl_problem_params = OptimalHarvestingLogisticProblem1D(T=1, B=1, y_initial=1, u_initial=-20, gamma=.006, r=10.0,
                                                          k=100.0, x_1_omega=0.3, x_2_omega=.4,  u_1=-10, u_2=10)


    def state_equation_function(u):
        def _state_equation_function(t, state):

            return ohl_problem_params.gamma * \
                   laplacian_operator_approximation_1d(state,
                                                       ohl_problem_params.h_state_space) + \
                   ohl_problem_params.r * state * \
                   (1 - state/ohl_problem_params.k) - \
                    ohl_problem_params.subspace_mask * u[int(t/ohl_problem_params.h_state_time)] * \
                   state

        return _state_equation_function

    def adjoint_state_equation_function(state, u):
        def _adjoint_state_equation_function(t, adjoint_state):
            # Використовуємо рівність A=A* для лапласівського оператора
            return -ohl_problem_params.gamma * \
                   laplacian_operator_approximation_1d(adjoint_state,
                                                       ohl_problem_params.h_state_space) - \
                   ohl_problem_params.r * adjoint_state * \
                   (1 - 2*state[int(t/ohl_problem_params.h_state_time)]/ohl_problem_params.k) + \
                    ohl_problem_params.subspace_mask * u[int(t/ohl_problem_params.h_state_time)] * \
                   (1 + adjoint_state)
        return _adjoint_state_equation_function

    def integrand_cost_function(state, adjoint_state, u):
        def _integrand_cost_function(t):
            return -np.sum(ohl_problem_params.subspace_mask *
                    u[int(t/ohl_problem_params.h_state_time)] *
                    state[int(t/ohl_problem_params.h_state_time)])
        return _integrand_cost_function

    def cost_derivative_u_function(u, state, adjoint_state):

        return state * (1 + adjoint_state)

    def projection_gradient_operator(u):
        u[u < ohl_problem_params.u_1] = ohl_problem_params.u_1

        u[u > ohl_problem_params.u_2] = ohl_problem_params.u_2
        return u

    optimal_harvesting_solver = PMPPDESolver(state_equation_function, adjoint_state_equation_function,
                                           integrand_cost_function, cost_derivative_u_function,
                                           projection_gradient_operator, 'Optimal_harvesting',
                                           ohl_problem_params.u_initial, ohl_problem_params.T,
                                           boundary_space=None, initial_state=ohl_problem_params.y_initial,
                                           eps_cost_derivative=ohl_problem_params.eps_cost_derivative,
                                           eps_gradient_step=ohl_problem_params.eps_steplength,
                                           init_gradient_step=ohl_problem_params.ro_init,
                                           gradient_adjustment=ohl_problem_params.b,
                                           time_grid_step=ohl_problem_params.h_state_time,
                                           space_grid_step=ohl_problem_params.h_state_space,
                                           gradient_step_max_iter=ohl_problem_params.max_ro)

    optimal_harvesting_solver.gradient_descent_loop()
    optimal_harvesting_solver.visualize_control()
    plt.show()
