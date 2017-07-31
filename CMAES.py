import numpy as np
from mpi4py import MPI
from collections import deque


class Optimizer():
    def __init__(self, rank, n_workers, N_dim=3, x_0=None, verbose=False):
            # adjust reinit() args if modifying init() args

        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.n_workers = n_workers
        self.verbose = verbose
        self.terminate = False
        self.error = False

        if n_workers > 1 and n_workers < 5 and rank==0:
            print 'Warning: Parallel mode with less than 5 workers is numerically unstable!'

        assert isinstance(N_dim, int), "N_dim ({}) must be integer!".format(N_dim)
        assert N_dim > 0, "N_dim ({}) must be positive!".format(N_dim) 
        self.N_dim = N_dim 

        if not isinstance(x_0, np.ndarray) and x_0:
            assert isinstance(x_0, np.ndarray), "x_0 ({}) must be ndarray!".format(x_0)
            assert x_0.dtype == float, "x_0 ({}) must be float array!".format(x_0)
            assert x_0.shape == (N_dim, ), "x_0 ({}) must match N_dim ({})!".format(x_0, N_dim)
            self.x_mean = x_0
        else:
            self.x_mean = np.random.randn(N_dim)

        self.x_0 = x_0
        self.x_recent = deque()
        self.x_recent.append(self.x_mean)
        self.sigma = 0.1                # coordinate-wise standard deviation (step-size)
        self.cost_stop = 1.       # stop if cost < stopcost (minimization)
        self.i_stop = 1e3 * N_dim**2    # stop after i_stop iterations
        self.cost_recent = deque([42.])
        self.ptp_stop = 0.01 #0.001
        self.max_reset = 100

        # Strategy parameter setting: Selection
        if self.n_workers == 1:
            self.lambda_ = int(4.0 + np.floor(3.0*np.log(N_dim)))  # recommended by Hansen
        else:
            self.lambda_ = self.n_workers -1      # population size, offspring number
        self.convergence_interval = int(10+np.ceil(30.0*N_dim/self.lambda_))       # window for convergence test
        self.mu_ = self.lambda_ / 2.0             # mu_ is float
        self.mu = int(np.floor(self.mu_))         # mu is integer = number of parents/points for recombination
        self.weights = np.zeros(self.mu)
        for i in xrange(self.mu):            # muXone recombination weights
            self.weights[i] = np.log(self.mu_+0.5) - np.log(i+1)
        self.weights /= sum(self.weights)         # normalize recombination weights array 
        self.mu_eff = sum(self.weights)**2 / sum(self.weights**2)
                                        # variance-effective size of mu

        # Strategy parameter setting: Adaptation
        self.c_c = (4.0+self.mu_eff/N_dim) / (N_dim+4.0+2*self.mu_eff/N_dim)
                                        # time constant for cumulation for C
        self.c_s = (self.mu_eff+2.0) / (N_dim+self.mu_eff+5.0)
                                        # time constant for cumulation for sigma control
        self.c_1 = 2.0 / ((N_dim+1.3)**2 + self.mu_eff) # learning rate for rank-one update of C
        self.c_mu = min(2 * (self.mu_eff-2.0+1.0/self.mu_eff) / ((N_dim+2.0)**2 + 2*self.mu_eff/2.0), 1.-self.c_1)
                                        # and for rank-mu update
        self.damps = 1.0 + 2*np.max([0, np.sqrt((self.mu_eff-1.0)/(N_dim+1.0))-1.0]) + self.c_s
                                        # damping for sigma

        # Initialize dynamic (internal) strategy parameters and constants
        self.p_c = np.zeros(N_dim)               # evolution path for C
        self.p_s = np.zeros(N_dim)               # evolution path for sigma
        self.B = np.eye(N_dim)                   # B defines the coordinate system
        self.D = np.eye(N_dim)
        self.B_D = np.dot(self.B, self.D)       # dot product of B and D
        self.C = np.dot(self.B_D, (self.B_D).T)               # covariance matrix
        if self.verbose and self.rank==0:
            print 'initializing'
            print 'lambda:', self.lambda_
            print 'mu:', self.mu
            print 'mu_eff:', self.mu_eff
            print 'c_c:', self.c_c
            print 'c_s:', self.c_s
            print 'c_1:', self.c_1
            print 'c_mu:', self.c_mu
            print 'self.c_1 + self.c_mu:', self.c_1 + self.c_mu
            print 'D:', self.D
            print 'B_D:', self.B_D
            print 'C:', self.C
        assert 1.0 - self.c_1 - self.c_mu >= 0.0, "Invalid adaptation rates of covariance!"
        self.i_eigen = 0                     # for updating B and D
        self.chi_N = np.sqrt(N_dim) * (1.0-1.0/(4.0*N_dim) + 1.0/(21.0*N_dim**2))
                                        # expectation of ||N(0,I)|| == norm(randn(N,1))

        # Initialize arrays
        self.z = np.zeros([self.lambda_, N_dim])
        self.x = np.zeros([self.lambda_, N_dim])
        self.cost = np.ones(self.lambda_)*1e9
        self.errors_all = np.zeros(self.lambda_, dtype='bool')


    def reinit(self):
        self.__init__(self.rank, self.n_workers, self.N_dim, self.x_0, self.verbose)


    def objective(self, x):     # objective function
        x_target = np.arange(1./(self.N_dim+1), 1., 1./(self.N_dim+1))
        #x_target = np.array(self.default_values) + np.random.randn(self.N_dim)
        deviation = x_target - x
        value = 100.0 * np.sqrt(np.dot(10.0*deviation, 0.1*deviation))
                                    # arbitrary objective function,
                                    #  minimal if x == xtarget
        return value


    def sampling(self, x_mean, sigma, B_D):

        z = np.random.randn(self.N_dim) # standard normally distributed vector
        x = x_mean + sigma * (np.dot(B_D, z))
                                    # add mutation, Eq. 37
        return x, z



    def terminate_process(self):
        print 'Master sending kill signal to workers'
        self.terminate = True
        for i_worker in xrange(1,self.n_workers):
            self.comm.send((None, None, None, True, self.error), dest=i_worker, tag=i_worker)
        return



    def run(self):
        if self.rank > 0:   # workers communicating with master
            while True:
                x_mean, sigma, B_D, terminate, error = self.comm.recv(source=0, tag=self.rank)
                if error:
                    print 'worker {} shutting down with error signal'.format(self.rank)
                    return True
                if terminate:
                    if self.verbose:
                        print 'worker {} shutting down'.format(self.rank)
                    return False

                x, z = self.sampling(x_mean, sigma, B_D)
                cost = self.objective(x)
                self.comm.send((x, z, cost, self.error), dest=0, tag=self.rank)

        else:   

        #######################################################
        # Master Generation Loop
        #######################################################

            self.x_all = []
            self.cost_all = []
            self.sigma_all = []
            self.i_reset = 0
            self.sigma0 = 0.1

            i_count = 0                     # the next 40 lines contain the 20 lines of interesting code
            while (self.i_reset < self.max_reset) and not self.terminate:

                if self.n_workers == 1:
                    # Generate and evaluate lambda offspring                    
                    for k in xrange(self.lambda_):
                        self.x[k] , self.z[k]= self.sampling(self.x_mean, self.sigma, self.B_D)
            
                        self.cost[k] = self.objective(self.x[k])
                                                # objective function call
                        i_count += 1
                    
                else:
                    for rank in xrange(1,self.n_workers):
                        self.comm.send((self.x_mean, self.sigma, self.B_D, self.terminate, self.error), dest=rank, tag=rank)
                    for rank in xrange(1,self.n_workers):
                        (self.x[rank-1], self.z[rank-1], self.cost[rank-1], self.errors_all[rank-1]) = self.comm.recv(source=rank, tag=rank)
                            # memory_all[0] is memory of master
                        if self.errors_all.any():
                            self.terminate = True
                            self.error = True
                            print 'Kill signal received. Shutting down.'
                            self.terminate_process()
                            return None, True
                        i_count += 1
                    #print 'i_count:', i_count


                # Sort by cost and compute weighted mean into x_mean
                indices = np.arange(self.lambda_)
                to_sort = zip(self.cost, indices)
                                            # minimization
                to_sort.sort()
                self.cost, indices = zip(*to_sort)
                self.cost = np.array(self.cost)
                indices = np.array(indices)
                index_max = indices[0] + 1  # offset accounts for master rank 0

                self.x_mean = np.zeros(self.N_dim)
                self.z_mean = np.zeros(self.N_dim)
                cost_mean = 0.0
                for i in xrange(self.mu):
                    self.x_mean += self.weights[i] * self.x[indices[i]]
                                            # recombination, Eq. 39
                    self.z_mean += self.weights[i] * self.z[indices[i]]
                                            # == D^-1 * B^T * (x_mean-x_old)/sigma
                    cost_mean += self.weights[i] * self.cost[indices[i]]
                
                # Cumulation: Update evolution paths
                self.p_s = (1.0-self.c_s)*self.p_s + (np.sqrt(self.c_s*(2.0-self.c_s)*self.mu_eff)) * np.dot(self.B,self.z_mean)
                                            # Eq. 40
                self.h_sig = int(np.linalg.norm(self.p_s) / np.sqrt(1.0-(1.0-self.c_s)**(2.0*i_count/self.lambda_))/self.chi_N < 1.4+2.0/(self.N_dim+1.0))
                self.p_c = (1.0-self.c_c)*self.p_c + self.h_sig * np.sqrt(self.c_c*(2.0-self.c_c)*self.mu_eff) * np.dot(self.B_D,self.z_mean)
                                            # Eq. 42

                # Adapt covariance matrix C               
                self.C = (1.0-self.c_1-self.c_mu)*self.C + self.c_1*(np.outer(self.p_c,self.p_c) + (1.0-self.h_sig)*self.c_c*(2.0-self.c_c)*self.C) + self.c_mu*np.dot(np.dot((np.dot(self.B_D, self.z[indices[:self.mu]].T)),np.diag(self.weights)),(np.dot(self.B_D, self.z[indices[:self.mu]].T)).T)
                                            # regard old matrix plus rank one update plus minor correction plus rank mu update, Eq. 43

                # Adapt step-size sigma
                self.sigma = self.sigma * np.exp((self.c_s/self.damps) * (np.linalg.norm(self.p_s)/self.chi_N - 1.0))
                                            # Eq. 41

                # Update B and D from C
                if i_count - self.i_eigen > self.lambda_/(self.c_1+self.c_mu)/self.N_dim/10.0:
                                            # to achieve O(N**2)
                    self.i_eigen = i_count
                    self.D, self.B = np.linalg.eigh(self.C) # eigen decomposition, B==normalized eigenvectors
                    self.D = np.diag(np.sqrt(self.D))
                                            # D contains standard deviations now
                    self.B_D = np.dot(self.B, self.D)

                # Escape flat cost, or better terminate?
                if self.cost[0] == self.cost[int(np.ceil(0.7*self.lambda_))-1]:
                    self.sigma *= np.exp(0.2+self.c_s/self.damps)
                    print 'warning: flat cost, consider reformulating the objective'
                    #print 'int(np.ceil(0.7*self.lambda_))-1:', int(np.ceil(0.7*self.lambda_))-1
                    #print 'cost:', self.cost

                while len(self.x_recent) > self.convergence_interval - 1:
                    self.x_recent.popleft()
                while len(self.cost_recent) > self.convergence_interval - 1:
                    self.cost_recent.popleft()
                self.x_recent.append(self.x_mean)
                self.cost_recent.append(cost_mean)
                x_min = self.x[0].copy()

                if ((np.ptp(self.x_recent, axis=0) < self.ptp_stop).all()) and len(self.x_recent)==self.convergence_interval:
                    if self.verbose:
                        print 'parameters converged. reinitializing search.'
                    self.i_reset += 1
                    self.reinit()
                    self.sigma = self.sigma0
                    if self.sigma0 < 0.5:
                        self.sigma0 += 0.05
                elif (np.ptp(self.cost_recent) < self.ptp_stop) and len(self.cost_recent)==self.convergence_interval:
                    if self.verbose:
                        print 'cost converged. reinitializing search.'
                    self.i_reset += 1
                    self.reinit()
                    self.sigma = self.sigma0
                    if self.sigma0 < 0.5:
                        self.sigma0 += 0.05
                if self.i_reset == self.max_reset:
                    print 'reinitialization limit reached. terminating search.'
                    self.terminate = True
                    self.terminate_process()
                    break

                self.x_all.append(self.x[0].copy())
                self.cost_all.append(self.cost[0])
                self.sigma_all.append(self.sigma)
                if self.verbose:
                    print '\n', i_count, ': cost', self.cost[0], '\nsigma:', self.sigma,\
                        '\nparams:', self.x[0]
            
                # Break if cost is good enough
                if self.cost[0] <= self.cost_stop:
                    print 'cost criterion reached (cost={}, cost_stop={})'.format(self.cost[0], self.cost_stop)
                    self.terminate = True
                    self.terminate_process()
                    break

                if i_count >= self.i_stop:
                    print 'Iteration limit reached. Terminating.'
                    self.terminate = True
                    self.terminate_process()
                    break


            print 'found solution (cost={}):'.format(self.cost[0])
            for i in xrange(self.N_dim):
                print '{}: {}'.format(i, x_min[i])

            return x_min, self.error



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_workers = comm.Get_size()
    rank = comm.Get_rank()

    myOptimizer = Optimizer(rank, n_workers, verbose=False)
    myOptimizer.run()
