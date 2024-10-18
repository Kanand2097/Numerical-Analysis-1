import numpy as np

def p1(data, eval_pts):
    """
    Implement the Lagrange interpolation method, and evaluate the interpolating polynomial 
    at the given points.

    @param data: a list of tuples [(x0, y0), (x1, y1), ..., (xn, yn)]
    @param eval_pts: a list of x values to evaluate the interpolating polynomial

    @return: a list of y values evaluated at the eval_pts
    """
    n = len(data)
    x_vals, y_vals = zip(*data)
    result = []
    
    for pt in eval_pts:
        total = 0
        for i in range(n):
            # Compute the Lagrange basis polynomial L_i(pt)
            L_i = 1
            for j in range(n):
                if i != j:
                    L_i *= (pt - x_vals[j]) / (x_vals[i] - x_vals[j])
            # Add to the total the term y_i * L_i(pt)
            total += y_vals[i] * L_i
        result.append(total)
    
    return np.array(result)
