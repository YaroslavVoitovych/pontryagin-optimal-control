import numpy as np


def runge_kutta_4order(t_prev, func_value_prev, func, h):
	k1 = func(t_prev, func_value_prev)
	k2 = func(t_prev + h/2, func_value_prev + k1*h/2)
	k3 = func(t_prev + h/2, func_value_prev + k2*h/2)
	k4 = func(t_prev + h, func_value_prev + k3*h)
	return func_value_prev + (k1 + 2*k2 + 2*k3 + k4)*h/6


def solve_ivp(right_side_function, initial_state, terminate_argument, discrete_param, backward=False):
	if backward:
		adjusted_right_side_function = lambda argument, state: -right_side_function(argument, state)
	else:
		adjusted_right_side_function = right_side_function
	states = [initial_state, ]
	state_prev = states[0]
	arg_prev = 0
	arg_space = np.arange(discrete_param, terminate_argument, discrete_param) if not backward else \
		np.arange(terminate_argument-2*discrete_param, -discrete_param, -discrete_param)
	# print(arg_space.shape)
	# print(list(map(int, arg_space/discrete_param)))
	for arg in arg_space:
		state_new = runge_kutta_4order(arg_prev, state_prev, adjusted_right_side_function, discrete_param)
		states.append(state_new)
		arg_prev = arg
		state_prev = state_new
	if backward:
		states = states[::-1]
	return np.array(states)

