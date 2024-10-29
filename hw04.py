import numpy as np

def p1(data, eval_pts):
    """
    Implement the divided difference method to interpolate the data points, 
    then evaluate the polynomial at the given points.

    @param data: a list of tuples [(x0, y0), (x1, y1), ..., (xn, yn)]
    @param eval_pts: a list of x values to evaluate the interpolating polynomial

    @return: a list of y values evaluated at the eval_pts
    """
    n = len(data)
    x_vals, y_vals = zip(*data)
    
    # Create a divided difference table
    div_diff = np.zeros((n, n))
    div_diff[:, 0] = y_vals
    
    for j in range(1, n):
        for i in range(n - j):
            div_diff[i, j] = (div_diff[i+1, j-1] - div_diff[i, j-1]) / (x_vals[i+j] - x_vals[i])

    # Evaluate the polynomial at the given eval_pts
    result = []
    for pt in eval_pts:
        total = div_diff[0, 0]
        prod_term = 1
        for k in range(1, n):
            prod_term *= (pt - x_vals[k-1])
            total += div_diff[0, k] * prod_term
        result.append(total)
    
    return np.array(result)

