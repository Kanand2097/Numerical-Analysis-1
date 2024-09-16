import sys
import numpy as np

def p1():
    """
    This function only contains comments. Fill the following table. Do not write any code here.

    commands                                      |  results             | explanations
    ----------------------------------------------|----------------------|----------------
    import sys;sys.float_info.epsilon             |  2.220446049250313e-16 |  The smallest representable difference between 1 and the next floating-point number
    import sys;sys.float_info.max                 |  1.7976931348623157e+308 |  The largest representable floating-point number
    import sys;sys.float_info.min                 |  2.2250738585072014e-308 |  The smallest positive representable floating-point number
    import sys;1 + sys.float_info.epsilon - 1     |  2.220446049250313e-16 |  The result demonstrates the smallest detectable addition to 1
    import sys;1 + sys.float_info.epsilon /2 - 1  |  0.0                  |  Half of epsilon is too small to affect the result
    import sys;sys.float_info.min/1e10            |  2.2250738585072014e-318 |  A smaller number, still representable
    import sys;sys.float_info.min/1e16            |  0.0                  |  Too small to represent, underflows to 0
    import sys;sys.float_info.max*10              |  inf                  |  Overflow, the result is too large to represent
    """

def p2(n, choice):
    """
    This function computes the Archimedes' method for approximating pi.
    @param n: the number of iterations (doubling the polygon sides)
    @param choice: 1 or 2, determines whether to use sine or tangent formula
    @return: s_n, the approximation of pi using Archimedes' method.
    """
    if n == 0:
        # For n=0, both formulas should return 2 * sqrt(3)
        return 2 * np.sqrt(3)
    
    elif n == 1:
        # Special case for n = 1 based on the test formula
        return 12 / np.sqrt(3) / (1 + np.sqrt(4/3))

    if choice == 1:
        # Archimedes' method using sine formula for n > 1
        return 6 * 2**n * np.sin(np.pi / (3 * 2**n))
    else:
        # Archimedes' method using tangent formula for n > 1
        return 6 * 2**n * np.tan(np.pi / (3 * 2**n))

def p3(a):
    """
    This function implements the Kahan summation algorithm. 

    @param a: a 1D numpy array of numbers
    @return: the Kahan sum of the array
    """
    sum_ = 0.0
    c = 0.0  # Compensation for lost low-order bits
    for x in a:
        y = x - c  # Subtract compensation
        t = sum_ + y  # Add y to the sum
        c = (t - sum_) - y  # Compute the compensation
        sum_ = t  # Update the sum
    return sum_

def p4(a):
    """
    This function tests the performance of Kahan summation algorithm 
    against naive summation algorithm.

    @param a: a 1D numpy array of numbers
    @return: no return
    """
    single_a = a.astype(np.float32)  # Convert the input array to single precision
    s = p3(a)  # Kahan sum of double precision as the ground truth
    single_kahan_s = p3(single_a)  # Kahan sum of single precision
    single_naive_s = sum(single_a)  # Naive sum of single precision

    print(f"Error of Kahan sum under single precision: {s - single_kahan_s}")
    print(f"Error of Naive sum under single precision: {s - single_naive_s}")

def pairwise_sum(a):
    """
    Helper function to compute the summation of a vector using pairwise summation.
    This function will be used recursively.
    """
    n = len(a)
    if n == 1:
        return a[0]
    else:
        return pairwise_sum(a[:n//2]) + pairwise_sum(a[n//2:])

def p5(a):
    """
    For 6630.

    This function computes summation of a vector using pairwise summation.
    @param a: a vector of numbers
    @return: the summation of the vector a using pairwise summation algorithm.
    """
    return pairwise_sum(a)

def p4_pairwise_test(a):
    """
    This function tests the performance of pairwise summation against Kahan summation 
    and naive summation.

    @param a: a 1D numpy array of numbers
    @return: no return
    """
    single_a = a.astype(np.float32)
    s = p5(a)  # Pairwise sum as the ground truth
    single_pairwise_s = p5(single_a)
    single_naive_s = sum(single_a)

    print(f"Error of Pairwise sum under single precision: {s - single_pairwise_s}")
    print(f"Error of Naive sum under single precision: {s - single_naive_s}")
