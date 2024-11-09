import numpy as np

def p1(data, powers):
    """
    Implement Richardson extrapolation to estimate the value at f(0).

    Parameters:
    - data: a list of values [f(2^(-1)), f(2^(-2)), ..., f(2^(-n))]
    - powers: a list of powers [alpha_1, alpha_2, ..., alpha_{n-1}]

    Returns:
    - Extrapolated value of f(0).
    """
    n = len(data)
    for i in range(len(powers)):
        new_data = []
        alpha = powers[i]
        for j in range(len(data) - 1):
            extrapolated_value = (2 ** alpha * data[j + 1] - data[j]) / (2 ** alpha - 1)
            new_data.append(extrapolated_value)
        data = new_data
    return data[0] if data else 0


def p2(beta):
    """
    Compute the value of the series sum_{k=0}^(infty) ((-1)^k /(2k + 1)^{beta}).

    Parameters:
    - beta: a real value for the parameter beta on (0, 1]

    Returns:
    - Approximate value of the series.
    """
    result = 0.0
    tolerance = 1e-10
    k = 0
    term = float('inf')
    while abs(term) > tolerance:
        term = ((-1) ** k) / (2 * k + 1) ** beta
        result += term
        k += 1
    return result


def p3(shifts):
    """
    Compute the coefficients of the finite difference scheme for f'(x).

    Parameters:
    - shifts: a list of real values (a_0, a_1, ..., a_n)

    Returns:
    - coefs: a numpy array of coefficients (c_0, c_1, ..., c_n).
    """
    n = len(shifts)
    matrix = np.array([[shift ** i for shift in shifts] for i in range(n)])
    rhs = np.zeros(n)
    rhs[1] = 1  # For the first derivative
    coefs = np.linalg.solve(matrix, rhs)
    return coefs

