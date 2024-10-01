import sys
import numpy as np

def p1(f, a, b, epsilon, name, f_prime=None):
    """
    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance
    @param name: the name of the method to use
    @param f_prime: the derivative of the function f (only needed for Newton's method)

    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    
    if name == "bisection":
        # Bisection method
        n = 0
        c = (a + b) / 2
        while abs(f(c)) > epsilon:
            n += 1
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
            c = (a + b) / 2
        return c, n

    elif name == "newton":
        # Newton's method
        if f_prime is None:
            raise ValueError("Newton's method requires the derivative of the function")
        n = 0
        c = a  # Initial guess (usually a or b)
        while abs(f(c)) > epsilon:
            n += 1
            c = c - f(c) / f_prime(c)
        return c, n

    elif name == "secant":
        # Secant method
        n = 0
        c_prev = a
        c_curr = b
        while abs(f(c_curr)) > epsilon:
            n += 1
            c_next = c_curr - f(c_curr) * (c_curr - c_prev) / (f(c_curr) - f(c_prev))
            c_prev, c_curr = c_curr, c_next
        return c_curr, n

    elif name == "regula_falsi":
        # False position method (Regula Falsi)
        n = 0
        c = a
        while abs(f(c)) > epsilon:
            n += 1
            c = a - f(a) * (b - a) / (f(b) - f(a))
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c, n

    elif name == "steffensen":
        # Steffensen's method
        n = 0
        c = a
        while abs(f(c)) > epsilon:
            n += 1
            g = f(c)
            c = c - g**2 / (f(c + g) - g)
        return c, n

    else:
        print("Invalid name")
        return None


def p2():
    """
    Summarize the iteration number for each method name in the table

    |name          | iter | 
    |--------------|------|
    |bisection     |      |
    |secant        |      |
    |newton        |      |
    |regula_falsi  |      |
    |steffensen    |      |
    """

def p3(f, a, b , epsilon):
    """
    For 6630 students only.

    Implement the Illinois algorithm to find the root of the function f in the interval [a, b]

    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance

    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    pass

def p4(f, a, b , epsilon):
    """
    For 6630 students only.

    Implement the Pegasus algorithm to find the root of the function f in the interval [a, b]

    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance
    
    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    pass
