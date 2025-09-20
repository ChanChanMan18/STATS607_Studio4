import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, ComputeStatistics

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
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
        


        
    


