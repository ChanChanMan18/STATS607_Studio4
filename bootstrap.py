import warnings
import numpy as np
import math

"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""

import numpy as np
import statsmodels.api as sm
import warnings

def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """

    # Checking that X and y are of correct shape
    if len(shape(X)) != 2:
        raise ValueError, f'X has dimension {len(shape(X))} instead of 2'
    if len(shape(y)) != 1:
        raise ValueError, f'y has dimension {len(shape(y))} instead of 1'
    if shape(X)[0] == 0 or shape(y)[0] == 0:
        raise ValueError, 'X or y is an empty array'
    if shape(X)[0] != shape(y)[0]:
        raise ValueError, 'Lengths of X and y do not match'

    # Setup
    n = np.shape(y)[0] # Length of dataset
    output = []

    # Main loop
    for _ in range(n_bootstrap):

        # Randomizing the bootstrapped indexes
        rng = np.random.default_rng()
        indexes = np.arange(n)
        random_index = rng.choice(indexes, size=n_bootstrap, replace=True)

        # Compute statistics on bootstrapped samples
        X_b = X[random_index]
        y_b = y[random_index]
        output.append(compute_stat(X_b, y_b))
        
    # Return
    return np.array(output)

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """

    # Check conditions on alpha
    if not isinstance(alpha, float):
        raise TypeError("alpha must be of type float")
    
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the interval (0, 1)")
    
    # Change into array if not one
    try:
        arr = np.asarray(bootstrap_stats)
    except Exception as message:
        raise TypeError("bootstrap_stats must be array-like") from message

    # Check size of array
    if arr.ndim == 0:
        raise ValueError("bootstrap_stats must contain at least one value")
    
    if arr.size == 0:
        raise ValueError("Input must have length greater than zero")
    
    # The entries of array must be floats
    try:
        arr = arr.astype(float)
    except Exception:
        raise TypeError("bootstrap_stats must be numeric")
        
    lower_bound = np.quantile(arr, alpha / 2.0)
    upper_bound = np.quantile(arr, 1.0 - alpha / 2.0)

    if lower_bound == upper_bound:
        warnings.warn("CI degenerated into point (lower == upper); bootstrap distribution may be discrete or too small", RuntimeWarning)

    return (lower_bound, upper_bound)

class ComputeStatistics:

    def R_squared(self, X, y):
        """
        Calculate R-squared from multiple linear regression.

        Parameters
        ----------
        X : array-like, shape (n, p+1)
            Design matrix
        y : array-like, shape (n,)

        Returns
        -------
        float
            R-squared value (between 0 and 1) from OLS
        
        Raises
        ------
        ValueError
            If X.shape[0] != len(y)
        """

        # Coerce to numpy + numeric
        try:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
        except Exception:
            raise TypeError("X and y must be numeric")

        # Check shape of inputs
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n, p + 1).")
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array of shape (n, ).")
        
        # Check compatible dimensions
        if (X.shape[0] != len(y)):
            raise ValueError("First dimension of design and output must be equal")
        
        if (X.shape[0] == 0 or X.shape[1] == 0):
            raise ValueError("X and y must be non-empty")
        
        # If output constant, R^2 undefined
        if np.allclose(y, y[0]):
            raise ValueError("R squared undefined for constant output")

        model = sm.OLS(y, X)
        results = model.fit()
        r_squared_statsmodels = results.rsquared

        return r_squared_statsmodels
    
    def mean_of_output(self, y):
        """
        Dummy statistic that computes mean of y

        Parameters
        ----------
        X : array-like, shape (n, p+1)
            Design matrix
        y : array-like, shape (n,)

        Returns
        -------
        float
            mean of output 
        
        """

        return np.mean(y)