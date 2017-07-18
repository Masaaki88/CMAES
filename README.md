# CMAES

This is a Python implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm. It's based on the Matlab version provided in Hansen's "The CMA Evolution Strategy: A Tutorial" (2011) [1] and the corresponding Wikipedia page [2]. It's written as a Python class in a MPI framework (mpi4py required [3]), which can be run both in a single-thread mode or parallel mode.

To use this algorithm for your own optimization problem, modify the objective() method accordingly, i.e. specify how input parameters x are mapped onto a single cost value, which needs to be minimized.

Finally, instantiate the object by providing N_dim (number of dimensions) and run() it.

2017/07/18  
written in Python 2.7.12  
by Max Murakami

[1] https://arxiv.org/abs/1604.00772  
[2] https://en.wikipedia.org/wiki/CMA-ES  
[3] http://pythonhosted.org/mpi4py/
