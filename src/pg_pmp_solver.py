import logging
import numpy as np
import matplotlib.pyplot as plt


from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.utils import l2_norm, viz_1d_control, viz_2d_heatmap
from src.ode_utils import solve_ivp


class PMPProjectedGradientSolver(ABC):
    def __init__(self, state_equation_function: Callable, adjoint_state_equation_function: Callable,
                 integrand_cost_function: Callable, cost_derivative_u_function: Callable,
                 projection_gradient_operator: Callable, problem_name: str, init_u: np.array,
                 terminate_time: int, boundary_space: np.array = None, initial_state:np.array=None,
                 eps_cost_derivative: np.float16 = 1e-2, eps_gradient_step: np.float16 = 1e-2,
                 init_gradient_step: np.float16 = 1.0,
                 gradient_adjustment: np.float16 = 0.6,
                 time_grid_step: np.float16 = 1e-2, space_grid_step: np.float16 = None,
                 gradient_step_max_iter: int = 20):

        self.logger = logging.getLogger('PMPSolver_logger')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.problem_name = problem_name
        self.logger.info(f'Ініціалізується задача {self.problem_name}')
        self.eps_cost_derivative = eps_cost_derivative
        self.eps_gradient_step = eps_gradient_step
        self.init_gradient_step = init_gradient_step
        self.gradient_adjustment = gradient_adjustment
        self.time_grid_step = time_grid_step
        self.terminate_time = terminate_time
        self.init_u = init_u
        self.boundary_space = boundary_space
        self.init_state = initial_state
        self.gradient_step_max_iter = gradient_step_max_iter
        self.logger.info(f'Порядок збіжності градієнта функції втрат: {self.eps_cost_derivative}')
        self.logger.info(f'Порядок збіжності кроку градієнта: {self.eps_gradient_step}')
        self.logger.info(f'Початковий крок градієнта: {self.init_gradient_step}')
        self.logger.info(f'Крок дискретизації по часу: {self.time_grid_step}')
        self.logger.info(f'Кінцеве значення часу: {self.terminate_time}')
        if space_grid_step is not None:
            self.space_grid_step = space_grid_step
            self.logger.info(f'Крок дискретизації по просторовій змінній: {self.space_grid_step}')
        self.logger.info(f'Початковий вектор керування: {self.init_u}')
        self.logger.info(f'Максимальна кількість ітерацій для пошуку найкращого кроку градієнтного спуску: {self.gradient_step_max_iter}')

        # Визнчимо функції (оператори) для рівнянь стану та спряженого стану
        self.state_equation_function = state_equation_function
        self.adjoint_state_equation_function = adjoint_state_equation_function
        self.cost_derivative_u_function = cost_derivative_u_function
        self.integrand_cost_function = integrand_cost_function
        self.gradient_projection_function = projection_gradient_operator

        # Визначаємо вектори часу та простору
        self.time_range = np.arange(0, self.terminate_time, self.time_grid_step)

        # Присвоєння початкових умов
        self.current_gradient_iteration = 0
        self.current_cost_derivative_u = np.array([np.inf])
        self.current_cost = np.inf
        self.new_cost = self.current_cost
        self.current_gradient_step = self.init_gradient_step
        self.current_gradient_step_iteration = 0
        self.current_u = self.init_u
        self.new_u = self.current_u

    def norm_gradient_stop_condition(self) -> bool:
        grad_norm = l2_norm(self.current_cost_derivative_u)

        return grad_norm < self.eps_cost_derivative

    def gradient_descent_loop_stop_condition(self) -> bool:
        stop_condition = (self.norm_gradient_stop_condition() &
                          (self.current_gradient_step < self.eps_gradient_step))
        return stop_condition

    @abstractmethod
    def visualize_control(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def solve_state_problem(self,  *args, **kwargs) -> np.array:
        pass

    @abstractmethod
    def solve_adjoint_state_problem(self, *args, **kwargs) -> np.array:
        pass

    @abstractmethod
    def integrate_cost(self, integrand_cost_function) -> np.float32:
        pass

    def adjust_gradient_step(self, *args, **kwargs) -> None:
        self.current_gradient_step *= self.gradient_adjustment

    def gradient_descent_loop(self) -> None:

        # На кожній ітерації перевірка умов збіжності алгоритму
        while not self.gradient_descent_loop_stop_condition():

            self.current_cost = self.new_cost

            # Розв'язок рівняння стану
            self.current_state = self.solve_state_problem(self.current_u)
            self.logger.info('State')
           # self.logger.info(self.current_state)
            # Розв'язок рівняння спряженого стану
            self.current_adjoint_state = self.solve_adjoint_state_problem(self.current_state, self.current_u)
            self.logger.info('adjoint State')
            #self.logger.info(self.current_adjoint_state)
            # Обчислення градієнита функції втрат з урахуванням аналітичної формули похідної по керуванню
            self.current_cost_derivative_u = self.cost_derivative_u_function(self.current_u, self.current_state,
                                                                             self.current_adjoint_state)

            # Перевірка умови на норму градієнта
            if self.norm_gradient_stop_condition():
                self.logger.info('Збіжності досягнуто')
                break

            self.logger.info(f'''Значення функції втрат: {self.current_cost}; 
                                значення l2-норми градієнта: {l2_norm(self.current_cost_derivative_u)};
                                ітерація №{self.current_gradient_iteration}.''')
            # Пошук кроку градієнтного спуску
            # На кожній ітерації виконується перевірка на величину поточного кроку та на приріст функції втрат
            self.current_gradient_step = self.init_gradient_step
            for i in range(0, self.gradient_step_max_iter):
                if self.current_gradient_step < self.eps_gradient_step:
                    break


                # Обчислення нового керування
                self.new_u = self.gradient_projection_function(self.current_u - self.current_gradient_step *
                                                              self.current_cost_derivative_u)
                print('cost_derivative')

                # Розв'язок нового рівняння стану
                self.new_state = self.solve_state_problem(self.new_u)

                # Розв'язок рівняння спряженого стану
                self.new_adjoint_state = self.solve_adjoint_state_problem(self.new_state, self.new_u)

                # Обчислення нового значення функції втрат
                cost_func = np.vectorize(self.integrand_cost_function(self.new_state, self.new_adjoint_state, self.new_u))
                self.new_cost = self.integrate_cost(cost_func)

                if self.new_cost >= self.current_cost:
                    self.adjust_gradient_step()
                else:
                    self.current_u = self.new_u
                    break
            else:
                self.current_u = self.new_u

            self.current_gradient_iteration += 1

            # Додаткова умова виходу, коли функція втрат збігається
            if np.abs(self.new_cost - self.current_cost) <= self.eps_cost_derivative:
                self.logger.info('Збіжності досягнуто')
                break
        else:
            self.logger.info('Збіжності досягнуто')


class PMPODESolver(PMPProjectedGradientSolver):
    def __init__(self, state_equation_function: Callable, adjoint_state_equation_function: Callable,
                 integrand_cost_function: Callable, cost_derivative_u_function: Callable,
                 projection_gradient_operator: Callable, problem_name: str, init_u: np.array,
                 terminate_time: int, boundary_space: np.array = None, initial_state: np.array=None,
                 eps_cost_derivative: np.float16 = 1e-3, eps_gradient_step: np.float16 = 1e-3,
                 init_gradient_step: np.float16 = 1.0, gradient_adjustment: np.float16 = 0.6,
                 time_grid_step: np.float16 = 1e-2, space_grid_step: np.float16 = None,
                 gradient_step_max_iter: int = 20):

        super().__init__(state_equation_function, adjoint_state_equation_function,
                 integrand_cost_function, cost_derivative_u_function,
                 projection_gradient_operator, problem_name, init_u,
                 terminate_time, boundary_space, initial_state,
                 eps_cost_derivative, eps_gradient_step,
                 init_gradient_step,gradient_adjustment,
                 time_grid_step, space_grid_step,
                 gradient_step_max_iter)

    def visualize_control(self):
        viz_1d_control(self.time_range, self.current_u)

    def solve_state_problem(self, u) -> np.array:
        state = solve_ivp(self.state_equation_function(u), self.init_state, self.terminate_time, self.time_grid_step)
        return state

    def solve_adjoint_state_problem(self, state, u) -> np.array:
        adjoint_state = solve_ivp(self.adjoint_state_equation_function(state, u), 0.0,
                                  self.terminate_time, self.time_grid_step, backward=True)
        return adjoint_state

    def integrate_cost(self, integrand_cost_function: Callable):
        return np.trapz(y=integrand_cost_function(self.time_range), x=self.time_range, dx=self.time_grid_step)


class PMPPDESolver(PMPProjectedGradientSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize_control(self, dimensions: int = 2) -> None:
        if dimensions == 2:
            print(self.current_u)
            viz_2d_heatmap(self.current_u)
            viz_2d_heatmap(self.current_state)
        else:
            viz_1d_control(self.time_range, self.current_u)

    def solve_state_problem(self, u) -> np.array:
        state = solve_ivp(self.state_equation_function(u), self.init_state, self.terminate_time, self.time_grid_step)
        return state

    def solve_adjoint_state_problem(self, state, u) -> np.array:
        adjoint_state = solve_ivp(self.adjoint_state_equation_function(state, u), np.zeros_like(self.init_state),
                                  self.terminate_time, self.time_grid_step, backward=True)
        return adjoint_state

    def integrate_cost(self, integrand_cost_function: Callable):
        return np.trapz(y=integrand_cost_function(self.time_range), x=self.time_range, dx=self.time_grid_step)


