import numpy as np


def runge_kutta_4order(t_prev, func_value_prev, func, h):
	k1 = func(t_prev, func_value_prev)
	k2 = func(t_prev + h/2, func_value_prev + k1*h/2)
	k3 = func(t_prev + h/2, func_value_prev + k2*h/2)
	k4 = func(t_prev + h, func_value_prev + k3*h)
	return func_value_prev + (k1 + 2*k2 + 2*k3 + k4)*h/6


