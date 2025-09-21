import pytest
import numpy as np
from scipy.stats import beta
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from bootstrap import bootstrap_sample, bootstrap_ci, ComputeStatistics

def test_bootstrap_integration(n = 1000, p = 3, sigma = .5, alpha = 0.05, compute_stats = ComputeStatistics.R_squared,
                               n_bootstrap_reps = 200):
    """
    Testing function that tests the the functions bootstrap_sample, bootstrap_ci, and
    the compute_stats function altogether.
    The number of bootstrap samples is taken equal to n, the sample size.
    Print out a tuple of two floats: (lower_bound, upper_bound) of the CI.

    Parameters
    ----------
    n : integer (default = 1000)
        sample size
    p : integer (default = 3)
        number of feature
    sigma : non-negative real number (default = 1)
        standard deviation of Gaussian error when design X and y
    alpha : real value in [0, 1) (default = 0.05)
        confidence interval threshold
    compute_stats : function in the class ComputeStatistics (default = ComputeStatistics.R_squared)
        function that computes the desired statistics 
    n_bootstrap_reps : int (default 200)
        Number of bootstrap repetitions to generate
    """
    print('============ Running integration test  ============')

    # Generate data
    X = np.hstack((
        np.random.rand(n, p),
        np.ones((n, 1))
    ))
    beta_star = np.random.rand(p+1, 1)
    y = X @ beta_star + np.random.rand(n, 1)*sigma
    y = y.reshape(-1)
    
    # Run bootstrapping
    bootstrap_stats = bootstrap_sample(X, y, compute_stats, n_bootstrap_reps, n)

    # Calculate bootstrap interval
    bootstrap_interval = bootstrap_ci(bootstrap_stats, alpha)
    print("    Obtained bootstrap CI: ", bootstrap_interval)
    print('===================================================')
    print()


def test_bootstrap_rsquared_vs_theory(n = 1000, p = 3, alpha = 0.05, n_bootstrap_reps = 200):
    """
    Testing function that tests the the functions bootstrap_sample, bootstrap_ci, and
    the ComputeStatistics.R_squared function altogether.
    The number of bootstrap samples is taken equal to n, the sample size.
    Print out two tuple, each of two floats:
        (lower_bound, upper_bound) of the bootstrapped CI.
        (lower_bound, upper_bound) of the theoretical CI, calculated using beta distribution.

    Parameters
    ----------
    n : integer (default = 1000)
        sample size
    p : integer (default = 3)
        number of feature
    alpha : real value in [0, 1) (default = 0.05)
        confidence interval threshold
    n_bootstrap_reps : int (default 200)
        Number of bootstrap repetitions to generate
    """
    print('========= Running rsquared vs theory test =========')

    # Generate data
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    # Run bootstrapping
    bootstrap_stats = bootstrap_sample(X, y, ComputeStatistics.R_squared, n_bootstrap_reps, n)

    # Calculate bootstrap interval
    bootstrap_interval = bootstrap_ci(bootstrap_stats, alpha)
    print("    Obtained bootstrap CI: ", bootstrap_interval)

    # Calculate theoretical bootstrap interval: Beta(p/2, (n-p-1)/2)
    quantiles = [alpha / 2.0, 1 - alpha / 2.0]
    left_ci, right_ci = beta.ppf(quantiles, p / 2, (n - p - 1) / 2)
    print("    Theoretical CI: ", (np.round(left_ci, 4), np.round(right_ci, 4)))
    print('===================================================')
    print()


def test_bootstrap_sample_returns_requested_length():
    """
    Test whether bootstrap sample returns the length inputted by user
    """
    rng = np.random.default_rng(82803)
    n = 50
    p = 3
    X = rng.normal(size=(n, p + 1))
    y = rng.normal(size = n)

    # Testing using the ComputeStatistics.mean_of_output statistics
    out = bootstrap_sample(X, 
                           y, 
                           ComputeStatistics.mean_of_output,
                           n_bootstrap_reps = 250,
                           n_bootstrap_samples = 250)
    assert isinstance(out, np.ndarray)
    assert out.shape == (250, )

    # Testing using the ComputeStatistics.R_squared statistics
    out = bootstrap_sample(X, 
                           y, 
                           ComputeStatistics.mean_of_output,
                           n_bootstrap_reps = 250,
                           n_bootstrap_samples = 250)
    assert isinstance(out, np.ndarray)
    assert out.shape == (250, )
    
def test_bootstrap_sample_bad_inputs():
    """
    Test on mismatched input dimensions
    """
    # Design bad input
    rng = np.random.default_rng(2)
    X = rng.normal(size = (10, 2))
    y_bad = rng.normal(size = 9) # mismatch y and design X

    # Testing unmatched y and design X using ComputeStatistics.mean_of_output statistics
    with pytest.raises(ValueError, match="do not match"):
        no_output = bootstrap_sample(X, 
                                  y_bad, 
                                  ComputeStatistics.mean_of_output, 
                                  n_bootstrap_reps = 10,
                                  n_bootstrap_samples = 10)

    # Testing unmatched y and design X using ComputeStatistics.R_squared statistics
    with pytest.raises(ValueError, match="do not match"):
        no_output = bootstrap_sample(X, 
                                  y_bad, 
                                  ComputeStatistics.R_squared, 
                                  n_bootstrap_reps = 10,
                                  n_bootstrap_samples = 10)
    
    # non-positive n_bootstrap_reps should throw error
    y = rng.normal(size = 10)
    with pytest.raises(ValueError, match="positive"):
        no_output = bootstrap_sample(X,
                                     y,
                                     ComputeStatistics.mean_of_output,
                                     n_bootstrap_reps = 0,
                                     n_bootstrap_samples = 10)

    # non-positive n_bootstrap_samples should throw error
    y = rng.normal(size = 10)
    with pytest.raises(ValueError, match="positive"):
        no_output = bootstrap_sample(X,
                                     y,
                                     ComputeStatistics.mean_of_output,
                                     n_bootstrap_reps = 10,
                                     n_bootstrap_samples = 0)
        

class TestCI:
    """
    Testing class for the `bootstrap_ci` function
    """

    def test_output_type(self):
        """Test on output type and shape"""
        bootstrap_stats = np.random.rand(100)*10.
        ci = bootstrap_ci(bootstrap_stats, 0.05)
        assert type(ci) == tuple, 'Output is not tuple'
        assert len(ci) == 2, 'Output size is not 2'

    def test_small_input(self):
        """Test on small input"""
        small_data = np.array([1, 2, 3]) # A small list of stats
        with pytest.warns(UserWarning, match="small"):
            bootstrap_ci(small_data, 0.05)

    def test_empty_cases(self):
        """Testing on empty input"""
        with pytest.raises(ValueError, match="greater than zero"):
            bootstrap_ci(np.array([], dtype='float'), alpha=0.1)

        with pytest.raises(ValueError, match="greater than zero"):
            bootstrap_ci(np.array([], dtype='float'), alpha=0.05)

        with pytest.raises(ValueError, match="greater than zero"):
            bootstrap_ci(np.array([], dtype='float'), alpha=0.01)

    def test_alpha_invalid(self):
        """Testing on invalid alpha"""
        bootstrap_stats = np.random.rand(100)*10.
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_ci(bootstrap_stats, 1.1)

        with pytest.raises(ValueError, match="alpha"):
            bootstrap_ci(bootstrap_stats, -0.5)

    def test_sort(self):
        """Testing against data with shuffled order"""
        bootstrap_stats = np.random.rand(20)*10.
        bootstrap_stats_increase = bootstrap_stats.copy()
        bootstrap_stats_increase.sort()
        bootstrap_stats_decrease = bootstrap_stats_increase[::-1]
        (l1, r1) = bootstrap_ci(bootstrap_stats, 0.05)
        (l2, r2) = bootstrap_ci(bootstrap_stats_increase, 0.05)
        (l3, r3) = bootstrap_ci(bootstrap_stats_decrease, 0.05)
        assert max(l1, l2, l3) - min(l1, l2, l3) < 1e-9, 'CI not consistent with respect to ordering'
        assert max(r1, r2, r3) - min(r1, r2, r3) < 1e-9, 'CI not consistent with respect to ordering'

class TestRSquared:
    """
    Testing class for the `R_squared` function
    """
    def test_empty_cases(self):
        """Testing on empty input"""
        with pytest.raises(ValueError, match='nonempty'):
            ComputeStatistics.R_squared(np.array([[], [], []], dtype='int'), np.array([1, 2, 3]))

        with pytest.raises(ValueError, match='nonempty'):
            ComputeStatistics.R_squared(np.array([[]], dtype='int'), np.array([], dtype='int'))

    def test_unequal_sample_size(self):
        """Testing on unequal sample size between X and y"""
        with pytest.raises(ValueError, match='equal'):
            ComputeStatistics.R_squared(np.array([[1, 2], [3, 5], [4, 9]]), np.array([1, 3, 5, 7]))

    def test_perfect_correlation(self):
        """Testing on a case where y is a linear combination of X"""
        X = np.array([[1, 1], [2, 4], [3, 3], [4, 6], [5, 8]])
        y = np.array([3, 8, 9, 14, 18])
        rs = ComputeStatistics.R_squared(X, y)
        assert np.abs(rs-1.) < 1e-9, f'Expected R squared of 1., got{rs}'


if __name__ == '__main__':
    test_bootstrap_integration(n = 1000, p = 3, sigma = .5, alpha = 0.05, compute_stats = ComputeStatistics.R_squared,
                               n_bootstrap_reps = 200)
    test_bootstrap_rsquared_vs_theory(n = 1000, p = 5, alpha = 0.2, n_bootstrap_reps = 1000)
