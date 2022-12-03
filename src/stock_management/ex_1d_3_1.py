import logging
import numpy as np
import matplotlib.pyplot as plt


from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.pg_pmp_solver import PMPODESolver
from src.utils import viz_1d_control

@dataclass
class StockManagementProblem:
    T: np.float16
    x_initial: np.float16
    u_1: np.float16
    u_2: np.float16
    u_initial: np.float16
    a: np.float16 = None
    c_1: np.float16 = None
    c_2: np.float16 = None
    g_1: np.float16 = None
    g_2: np.float16 = None
    g_3: np.float16 = None
    g_4: np.float16 = None
    h_state: np.float16 = 1e-2
    eps_cost_derivative: np.float16 = 1e-5
    ro_init: np.float16 = 1
    b: np.float16 = 0.6
    eps_steplength: np.float16 = 1e-2
    max_ro: np.int16 = 20
    max_iter: np.int16 = 100

    def __post_init__(self):
        # self.g_t = np.vectorize(lambda t: self.g_1*t + self.g_2 if self.T/2 >= t >= 0 else self.g_3 - self.g_4 * t)
        # self.psi_x = np.vectorize(lambda x: self.c_1 * x ** 2 if x >= 0 else self.c_2 * x ** 2)
        # self.psi_x_derivative = np.vectorize(lambda x: 2 * self.c_1 * x if x >= 0 else 2 * self.c_2 * x)
        self.u_initial *= np.ones(shape=(int(self.T/self.h_state)))


def analytic_solution_control(t):
    return -(np.exp(1-t) - np.exp(-1+t))/(np.exp(1) + np.exp(-1))


def analytic_solution_state(t):
    return (np.exp(1-t) + np.exp(-1+t))/(np.exp(1) + np.exp(-1))


if __name__ == '__main__':
    sm_problem_params = StockManagementProblem(T=1.0, x_initial=1.0, u_1=-np.inf, u_2=np.inf, u_initial=0.0)

    def state_equation_function(u):
        def _state_equation_function(t, state):
            return u[int(t/sm_problem_params.h_state)]
        return _state_equation_function

    def adjoint_state_equation_function(state, u):
        def _adjoint_state_equation_function(t, adjoint_state):
            return -state[int(t/sm_problem_params.h_state)]
        return _adjoint_state_equation_function

    def integrand_cost_function(state, adjoint_state, u):
        def _integrand_cost_function(t):
            return 0.5*(np.power(u[int(t/sm_problem_params.h_state)], 2) + np.power(state[int(t/sm_problem_params.h_state)],2))
        return _integrand_cost_function

    def cost_derivative_u_function(u, state, adjoint_state):
        return adjoint_state + u


    def projection_gradient_operator(u):
        def _operator(_u):
            if sm_problem_params.u_1 <= _u <= sm_problem_params.u_2:
                return _u
            elif _u < sm_problem_params.u_1:
                return sm_problem_params.u_1
            else:
                return sm_problem_params.u_2

        if isinstance(u, np.ndarray):
            return np.vectorize(_operator)(u)
        else:
            return _operator(u)


    ex1_solver = PMPODESolver(state_equation_function, adjoint_state_equation_function,
                                           integrand_cost_function, cost_derivative_u_function,
                                           projection_gradient_operator, 'Exercise 3.1',
                                           sm_problem_params.u_initial, sm_problem_params.T,
                                           boundary_space=None, initial_state=sm_problem_params.x_initial,
                                           eps_cost_derivative=sm_problem_params.eps_cost_derivative,
                                           eps_gradient_step=sm_problem_params.eps_steplength,
                                           init_gradient_step=sm_problem_params.ro_init,
                                           gradient_adjustment=sm_problem_params.b,
                                           time_grid_step=sm_problem_params.h_state, space_grid_step=None,
                                           gradient_step_max_iter=sm_problem_params.max_ro)

    ex1_solver.gradient_descent_loop()
    ex1_solver.visualize_control()
    viz_1d_control(ex1_solver.time_range, analytic_solution_control(ex1_solver.time_range))
    viz_1d_control(ex1_solver.time_range, analytic_solution_state(ex1_solver.time_range))
    print('Аналітичний розвязок, мінімум функціоналу',
          np.trapz(y=0.5*(analytic_solution_state(ex1_solver.time_range)**2 + analytic_solution_control(ex1_solver.time_range)**2), x=ex1_solver.time_range, dx=ex1_solver.time_grid_step))
    plt.show()
