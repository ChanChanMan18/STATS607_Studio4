# STATS 607 Studio 4 implementation

Khoa and Chandler's implemetation of Studio 4.
Use `python3 -m pytest test_bootstrap.py` to run pytest on the testcases.
Use `python3 test_bootstrap.py` to run the test file itself, which only runs the two integration tests.
The theoretical CI appears to be lower than the bootstrap CI. We suspect that this is because when sampling the same number of bootstrap samples as the sample size, some rows are taken more than once and hence the r2 increases.