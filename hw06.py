import numpy as np
import matplotlib.pyplot as plt

# Problem 1: Composite Quadrature Rules
def p1(func, a, b, n, option):
    """
    Implement composite quadrature rules for numerical integration
    of a function over the interval [a, b] with n subintervals.
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n

    if option == 1:  # Midpoint Rule
        midpoints = (x[:-1] + x[1:]) / 2
        ret = h * np.sum(func(midpoints))

    elif option == 2:  # Trapezoidal Rule
        ret = h * (0.5 * func(a) + np.sum(func(x[1:-1])) + 0.5 * func(b))

    elif option == 3:  # Simpson's Rule
        if n % 2 != 0:
            n += 1  # Adjust n to the next even number
            x = np.linspace(a, b, n + 1)
            h = (b - a) / n
        ret = h / 3 * (
            func(a)
            + 2 * np.sum(func(x[2:-1:2]))
            + 4 * np.sum(func(x[1::2]))
            + func(b)
        )
    else:
        raise ValueError("Invalid option value. Must be 1, 2, or 3.")
    return ret


# Problem 3: Romberg Integration
def p3(func, a, b, N, option):
    """
    Implement Romberg integration using the implemented quadrature rules.
    """
    R = np.zeros((N + 1, N + 1))  # Initialize Romberg table
    for k in range(N + 1):
        n = 2**k  # Number of subintervals doubles each step
        R[k, 0] = p1(func, a, b, n, option)

    # Generate Richardson extrapolation powers dynamically
    if option in [1, 2]:  # Midpoint or Trapezoidal Rule
        powers = [4**j for j in range(1, N + 1)]  # Error reduces as O(h^2), O(h^4), etc.
    elif option == 3:  # Simpson's Rule
        powers = [16**j for j in range(1, N + 1)]  # Error reduces as O(h^4), O(h^6), etc.

    for j in range(1, N + 1):
        for k in range(j, N + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (powers[j - 1] - 1)

    return R[N, N]


# Problem 4: Gauss Quadrature Rule
def p4():
    """
    Construct the Gauss quadrature rule using the roots of the Legendre polynomial of degree 6.
    """
    from numpy.polynomial.legendre import leggauss
    roots, weights = leggauss(6)
    return np.column_stack((roots, weights))


# Problem 5: General Gauss Quadrature (Optional for 6630)
def p5(n):
    """
    For 6630 ONLY: Construct the Gauss quadrature rule using the roots of the Legendre polynomial of degree n.
    """
    from numpy.polynomial.legendre import leggauss
    roots, weights = leggauss(n)
    return np.column_stack((roots, weights))


###############################################################################
#                                                                             #
# Helper functions for Problem 4 and 5, do not modify these functions         #
#                                                                             #
###############################################################################

# Helper function for p4
def legendre_poly_6(x):
    """
    Evaluate the Legendre polynomial of degree 6 at x.
    
    @param x: The value at which to evaluate the Legendre polynomial.
    @return: The value of the Legendre polynomial of degree 6 at x.
    """
    return (231 * x**6 - 315 * x**4 + 105 * x**2 - 5) / 16


# Helper functions for p5
def legendre_poly(n, x):
    """
    Evaluate the Legendre polynomial of degree n at x.
    
    @param n: The degree of the Legendre polynomial to evaluate.
    @param x: The value at which to evaluate the Legendre polynomial.
    @return: The value of the Legendre polynomial of degree n at x.
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) / n * x * legendre_poly(n - 1, x)) - ((n - 1) / n * legendre_poly(n - 2, x))


def deriv_legendre_poly(n, x):
    """
    Evaluate the derivative of the Legendre polynomial of degree n at x.
    
    @param n: The degree of the Legendre polynomial whose derivative to evaluate.
    @param x: The value at which to evaluate the derivative of the Legendre polynomial.
    @return: The value of the derivative of the Legendre polynomial of degree n at x.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return n / (x**2 - 1) * (x * legendre_poly(n, x) - legendre_poly(n - 1, x))

