import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, ComputeStatistics

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""

    # Generate data

    # Run bootstrapping

    # Calculate bootstrap interval

    # Calculate theoretical bootstrap interval
    
    # Return
    pass



"""
These functions test the function bootstrap_sample

"""
def test_bootstrap_sample_returns_requested_length():
    rng = np.random.default_rng(82803)
    n = 50
    p = 3
    X = rng.normal(size=(n, p + 1))
    y = rng.normal(size = n)

    compute_stats = ComputeStatistics()
    out = bootstrap_sample(X, 
                           y, 
                           compute_stats.mean_of_output,
                           n_bootstrap = 250)
    
    assert isinstance(out, np.ndarray)
    assert out.shape == (250, )
    
def test_bootstrap_sample_bad_inputs():
    rng = np.random.default_rng(2)
    X = rng.normal(size = (10, 2))
    y_bad = rng.normal(size = 9) # mismatch y and design X

    compute_stats = ComputeStatistics()
    with pytest.raises(ValueError, "X and y must have correct dimensions"):
        no_output = bootstrap_sample(X, 
                                  y_bad, 
                                  compute_stats.mean_of_output, 
                                  n_bootstrap = 10)
    
    # non-positive n_bootstrap should throw error
    y = rng.normal(size = 10)
    with pytest.raises(ValueError, "Must have at least one bootstrapped sample"):
        no_output = bootstrap_sample(X,
                                     y,
                                     compute_stats.mean_of_output,
                                     n_bootstrap = 0)
        


        
    


class TestCI:
    """
    Testing class for the `bootstrap_ci` function
    """

    def test_output_type(self):
        """Test on output type and shape"""
        bootstrap_stats = np.random.rand(100)*10.
        ci = bootstrap_ci(bootstrap_stats, 0.05)
        assert type(ci) == tuple 'Output is not tuple'
        assert len(ci) == 2, 'Output size is not 2'

    def test_small_input(self):
        """Test on small input"""
        small_data = np.array([1, 2, 3]) # A small list of stats
        with pytest.warns(UserWarning, match="small"):
            bootstrap_ci(small_data, 0.05)

    def test_empty_cases(self):
        """Testing on empty input"""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]), alpha=0.1)

        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]), alpha=0.05)

        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]), alpha=0.01)

    def test_alpha_invalid(self):
        """Testing on invalid alpha"""
        bootstrap_stats = np.random.rand(100)*10.
        with pytest.raise(ValueError, match="alpha"):
            bootstrap_ci(bootstrap_stats, 1.1)

        with pytest.raise(ValueError, match="alpha"):
            bootstrap_ci(bootstrap_stats, -0.5)

    def test_sort(self):
        """Testing against data with shuffled order"""
        bootstrap_stats = np.random.rand(1000)*10.
        bootstrap_stats_increase = bootstrap_stats.copy().sort()
        bootstrap_start_decrease = bootstrap_stats.copy().sort()[::-1]
        (l1, r1) = bootstrap_ci(bootstrap_stats, 0.05)
        (l2, r2) = bootstrap_ci(bootstrap_stats_increase, 0.05)
        (l3, r3) = bootstrap_ci(bootstrap_stats_decrease, 0.05)
        assert max(l1, l2, l3) - min(l1, l2, l3) > 1e-9, 'CI not consistent with respect to ordering'
        assert max(r1, r2, r3) - min(r1, r2, r3) > 1e-9, 'CI not consistent with respect to ordering'

class TestRSquared:
    """
    Testing class for the `R_squared` function
    """
    def test_empty_cases(self):
        """Testing on empty input"""
        with pytest.raises(ValueError, match='empty'):
            r_squared(np.array([[], [], []]), np.array([1, 2, 3]))

        with pytest.raises(ValueError, match='empty'):
            r_squared(np.array([]), np.array([]))

    def test_unequal_sample_size(self):
        """Testing on unequal sample size between X and y"""
        with pytest.raise(ValueError, mach='unequal'):
            r_squared(np.array([[1, 2], [3, 5], [4, 9]]), np.array([1, 3, 5, 7]))

    def test_perfect_correlation(self):
        """Testing on a case where y is a linear combination of X"""
        X = np.array([[1, 2, 3, 4, 5], [1, 4, 3, 6, 8]])
        y = np.array([3, 8, 9, 14, 18])
        rs = r_squared(X, y)
        assert np.abs(rs-1.) > 1e-9, f'Expected R squared of 1., got{rs}'





