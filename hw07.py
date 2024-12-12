import numpy as np

def p1(func, y0, tspan, n_steps, method):
    """
    Solve the ODE y' = f(t, y) using the specified method (euler, midpoint, rk4).
    
    @param func: The function f(t, y).
    @param y0: The initial condition y(0).
    @param tspan: The time span [t0, tf].
    @param n_steps: The number of time steps to take.
    @param method: The method to use to solve the ODE.
    @return: The solution array to the ODE at each time step.
    """
    h = (tspan[1] - tspan[0]) / n_steps
    t = np.linspace(tspan[0], tspan[1], n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    if method == 'euler':
        for i in range(n_steps):
            y[i + 1] = y[i] + h * func(t[i], y[i])
    elif method == 'midpoint':
        for i in range(n_steps):
            k1 = h * func(t[i], y[i])
            k2 = h * func(t[i] + h / 2, y[i] + k1 / 2)
            y[i + 1] = y[i] + k2
    elif method == 'rk4':
        for i in range(n_steps):
            k1 = h * func(t[i], y[i])
            k2 = h * func(t[i] + h / 2, y[i] + k1 / 2)
            k3 = h * func(t[i] + h / 2, y[i] + k2 / 2)
            k4 = h * func(t[i] + h, y[i] + k3)
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        raise ValueError("Invalid method. Choose 'euler', 'midpoint', or 'rk4'.")

    return y
