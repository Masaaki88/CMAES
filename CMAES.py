import matplotlib
matplotlib.use('Agg')                  # for use on clusters
from matplotlib import pyplot as plt
import numpy as np
import os
import rate_analysis
import ratio_analysis
from mpi4py import MPI
from collections import deque
import sys
import traceback
import cPickle
import subprocess
import shutil
import aux
from model2.model2 import Agent
from analysis import analyze_model2


class Optimizer():
    def __init__(self, name, target_rates, target_diffs, fitting_mode,
        failure_rates, group, fitness_stop, rank, 
        n_workers, outputpath, model=2, verbose=False, memory_profiler=None, 
        memory_profiling_file=None,
        temperature=None, pred_tau=None, ssc_step_decay=None, ssc_step_decay_im=None,
        init_sal_discs=None, init_sal_white=None, sal_image_max=None, retention_factor=None,
        fixation_cycle=None, sigma_lr=None, debug_mode=False):
            # adjust reinit() args if modifying init() args

        if rank==0:
            print 'Initializing Optimizer for subject {}.'.format(name)       
            if verbose:
                print 'target_rates:', target_rates
                print 'target_diffs:', target_diffs
                print 'failure_rates:', failure_rates
                print 'fitness_stop:', fitness_stop
                print 'outputpath:', outputpath
                print 'fiting_mode:', fitting_mode
                print 'debug_mode:', debug_mode
                print 'model:', model

        self.comm = MPI.COMM_WORLD
        self.memory_profiler = memory_profiler
        self.memory_profiling_file = memory_profiling_file

        self.name = name
        self.rank = rank
        self.n_workers = n_workers
        self.verbose = verbose
        self.terminate = False
        self.error = False
        self.params_folder = '{}params/'.format(outputpath)
        self.outputpath = outputpath
        if verbose:
            print "worker {}'s outputpath: {}".format(rank, self.outputpath)
            print "worker {}'s params_folder: {}".format(rank, self.params_folder)
        self.group = group
        self.fitting_mode = fitting_mode
        self.debug_mode = debug_mode

        if model not in [1,2]:
            self.model = 2
            print 'model not recognized. Simulating model 2.'
        else:
            self.model = model

        N_dim = 0
        i_value = 0
        self.values_all = []
        self.default_values = []
        self.value_names_fit = []
        self.value_names_fit_dic = {}
        default_stds = []

        self.temperature_init = temperature
        if temperature:
            self.temperature_def = temperature
        else:
            self.temperature_def = 0.6
            self.default_values.append(self.temperature_def)
            temperature_std = 0.1
            default_stds.append(temperature_std)
            self.value_names_fit.append('temperature')
            self.value_names_fit_dic.update({'temperature':i_value})
            N_dim += 1
        self.values_all.append(self.temperature_def)
        i_value += 1
        self.pred_tau_init = pred_tau
        if pred_tau:
            self.pred_tau_def = pred_tau
        else:
            self.pred_tau_def = 0.8
            self.default_values.append(self.pred_tau_def)
            pred_tau_std = 0.1
            default_stds.append(pred_tau_std)
            self.value_names_fit.append('pred_tau')
            self.value_names_fit_dic.update({'pred_tau':i_value})
            N_dim += 1
        self.values_all.append(self.pred_tau_def)
        i_value += 1
        self.ssc_step_decay_init = ssc_step_decay
        if ssc_step_decay:
            self.ssc_step_decay_def = ssc_step_decay
        else:  
            self.ssc_step_decay_def = 0.8
            self.default_values.append(self.ssc_step_decay_def)
            ssc_step_decay_std = 0.1
            default_stds.append(ssc_step_decay_std)
            self.value_names_fit.append('ssc_step_decay')
            self.value_names_fit_dic.update({'ssc_step_decay':i_value})
            N_dim += 1
        self.values_all.append(self.ssc_step_decay_def)
        i_value += 1
        self.ssc_step_decay_im_init = ssc_step_decay_im
        if ssc_step_decay_im:
            self.ssc_step_decay_im_def = ssc_step_decay_im
        else:
            self.ssc_step_decay_im_def = 0.8
            self.default_values.append(self.ssc_step_decay_im_def)
            ssc_step_decay_im_std = 0.1
            default_stds.append(ssc_step_decay_im_std)
            self.value_names_fit.append('ssc_step_decay_im')
            self.value_names_fit_dic.update({'ssc_step_decay_im':i_value})
            N_dim += 1
        self.values_all.append(self.ssc_step_decay_im_def)
        i_value += 1
        self.init_sal_discs_init = init_sal_discs
        if init_sal_discs:
            self.init_sal_discs_def = init_sal_discs
        else:
            self.init_sal_discs_def = 0.1
            self.default_values.append(self.init_sal_discs_def)
            init_sal_discs_std = 0.05
            default_stds.append(init_sal_discs_std)
            self.value_names_fit.append('init_sal_discs')
            self.value_names_fit_dic.update({'init_sal_discs':i_value})
            N_dim += 1
        self.values_all.append(self.init_sal_discs_def)
        i_value += 1
        self.init_sal_white_init = init_sal_white
        if init_sal_white:
            self.init_sal_white_def = init_sal_white
        else:
            self.init_sal_white_def = 0.01
            self.default_values.append(self.init_sal_white_def)
            init_sal_white_std = 0.005
            default_stds.append(init_sal_white_std)
            self.value_names_fit.append('init_sal_white')
            self.value_names_fit_dic.update({'init_sal_white':i_value})
            N_dim += 1
        self.values_all.append(self.init_sal_white_def)
        i_value += 1
        self.sal_image_max_init = sal_image_max
        if sal_image_max:
            self.sal_image_max_def = sal_image_max
        else:
            self.sal_image_max_def = 0.5
            self.default_values.append(self.sal_image_max_def)
            sal_image_max_std = 0.1
            default_stds.append(sal_image_max_std)
            self.value_names_fit.append('sal_image_max')
            self.value_names_fit_dic.update({'sal_image_max':i_value})
            N_dim += 1
        self.values_all.append(self.sal_image_max_def)
        i_value += 1
        self.retention_factor_init = retention_factor
        if retention_factor:
            self.retention_factor_def = retention_factor
        else:
            self.retention_factor_def = 0.9
            self.default_values.append(self.retention_factor_def)
            retention_factor_std = 0.1
            default_stds.append(retention_factor_std)
            self.value_names_fit.append('retention_factor')
            self.value_names_fit_dic.update({'retention_factor':i_value})
            N_dim += 1
        self.values_all.append(self.retention_factor_def)
        i_value += 1
        self.fixation_cycle_init = fixation_cycle
        if fixation_cycle:
            self.fixation_cycle_def = fixation_cycle
        else:
            self.fixation_cycle_def = 0.5
            self.default_values.append(self.fixation_cycle_def)
            fixation_cycle_std = 0.1
            default_stds.append(fixation_cycle_std)
            self.value_names_fit.append('fixation_cycle')
            self.value_names_fit_dic.update({'fixation_cycle':i_value})
            N_dim += 1
        self.values_all.append(self.fixation_cycle_def)
        i_value += 1
        self.sigma_lr_init = sigma_lr
        if sigma_lr:
            self.sigma_lr_def = sigma_lr
        else:
            self.sigma_lr_def = 100.0
            self.default_values.append(self.sigma_lr_def)
            sigma_lr_std = 10.0
            default_stds.append(sigma_lr_std)
            self.value_names_fit.append('sigma_lr')
            self.value_names_fit_dic.update({'sigma_lr':i_value})
            N_dim += 1
        self.values_all.append(self.sigma_lr_def)
        i_value += 1

        '''
        temperature_std = 0.1
        self.pred_tau_def = 0.85
        pred_tau_std = 0.1
        self.ssc_step_decay_def = 0.95
        ssc_step_decay_std = 0.1
        self.ssc_step_decay_im_def = 0.995
        ssc_step_decay_im_std = 0.1
        self.init_sal_discs_def = 0.1
        init_sal_discs_std = 0.01
        self.init_sal_white_def = 0.01
        init_sal_white_std = 0.001
        self.sal_image_max_def = 0.55
        sal_image_max_std = 0.1
        self.retention_factor_def = 0.9
        retention_factor_std = 0.1
        self.fixation_cycle_def = 0.5
        fixation_cycle_std = 0.1
        self.sigma_lr_def = 100.0
        sigma_lr_std = 10.0
        '''
        self.default_values_all = [self.temperature_def, self.pred_tau_def, self.ssc_step_decay_def, 
            self.ssc_step_decay_im_def, self.init_sal_discs_def, self.init_sal_white_def, 
            self.sal_image_max_def, self.retention_factor_def, self.fixation_cycle_def, self.sigma_lr_def]
        self.value_names_all = ['temperature', 'pred_tau', 'ssc_step_decay', 'ssc_step_decay_im', 
            'init_sal_discs', 'init_sal_white', 'sal_image_max', 'retention_factor', 'fixation_cycle', 'sigma_lr']

        self.target_rates = target_rates  
        self.target_diffs = target_diffs
        self.failure_rates = failure_rates
        self.N_sessions = 3
        for i_session in xrange(3):
            if (target_rates[i_session] < 0.0).all():
                self.N_sessions -= 1
        self.N_dim = N_dim  
        self.sigma = 0.1                # coordinate-wise standard deviation (step-size)
        self.fitness_stop_def = fitness_stop       # stop if fitness < stopfitness (minimization)
        self.fitness_stop = self.fitness_stop_def * self.N_sessions
        if verbose:
            print 'fitness_stop:', self.fitness_stop
        self.i_stop = 1e3 * N_dim**2    # stop after i_stop iterations
        #self.x_mean = np.random.randn(N_dim)    # objective variables initial point
        #x_init = np.array([0.5, 0.5])  # initial PRED_TAU and TEMPERATURE
        #x_init = np.ones(N_dim)
        #for i in xrange(N_dim):
        #    x_init[i] = self.default_values[i]
        #self.x_mean = self.sigma * np.random.randn(N_dim) + x_init
        self.x_mean = self.default_values
        self.x_recent = deque()
        self.x_recent.append(self.x_mean)
        self.fitness_recent = deque([42.])
        self.ptp_stop = 0.01 #0.001
        self.max_reset = 100

        # Strategy parameter setting: Selection
        #self.lambda_ = int(4.0 + np.floor(3.0*np.log(N_dim)))
        self.lambda_ = n_workers -1
                                        # population size, offspring number
        self.convergence_interval = int(10+np.ceil(30.0*N_dim/self.lambda_))       # window for convergence test
        if self.verbose and self.rank==0:
            print 'convergence interval:', self.convergence_interval
        self.mu_ = self.lambda_ / 2.0             # mu_ is float
        self.mu = int(np.floor(self.mu_))         # mu is integer = number of parents/points for recombination
        self.weights = np.zeros(self.mu)
        for i in xrange(self.mu):            # muXone recombination weights
            self.weights[i] = np.log(self.mu_+0.5) - np.log(i+1)
        self.weights /= sum(self.weights)         # normalize recombination weights array 
        self.mu_eff = sum(self.weights)**2 / sum(self.weights**2)
                                        # variance-effective size of mu
        if debug_mode:
            self.fitness_stop = .1
            self.ptp_stop = 0.1
            self.i_stop = 20
            self.max_reset = 2
            self.convergence_interval = 5

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
        #self.D = np.eye(N_dim)                   # diagonal matrix D defines the scaling
        #self.D = np.diag([temperature_std, pred_tau_std, ssc_step_decay_std, 
        #    ssc_step_decay_im_std, init_sal_discs_std, init_sal_white_std, 
        #    sal_image_max_std, retention_factor_std, fixation_cycle_std, sigma_lr_std])
        #self.D = np.diag(self.default_values)
        self.D = np.diag(default_stds)
        self.B_D = np.dot(self.B, self.D)
        self.C = np.dot(self.B_D, (self.B_D).T)               # covariance matrix
        if self.verbose and self.rank==1:
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
        self.fitness = np.ones(self.lambda_)*1e9
        self.rates = np.zeros([self.lambda_, 3, 2])
        self.diffs = np.zeros([self.lambda_, 3])
        self.corrupts = np.zeros(self.lambda_, dtype='int')
        self.errors_all = np.zeros(self.lambda_, dtype='bool')
        self.memory_all = np.zeros(self.lambda_+1)  # also includes memory of master


    def reinit(self):
        self.__init__(self.name, self.target_rates, self.target_diffs,
            self.fitting_mode, self.failure_rates, 
            self.group, self.fitness_stop_def, self.rank, self.n_workers, self.outputpath, self.model,
            self.verbose, self.memory_profiler, self.memory_profiling_file, self.temperature_init, 
            self.pred_tau_init, self.ssc_step_decay_init, self.ssc_step_decay_im_init,
            self.init_sal_discs_init, self.init_sal_white_init, self.sal_image_max_init,
            self.retention_factor_init, self.fixation_cycle_init, self.sigma_lr_init, self.debug_mode)


    def write_params(self):
        if self.verbose and self.rank==1:
            print 'worker {} writing parameters.'.format(self.rank)

        '''
        temperature = self.temperature_def
        pred_tau = self.pred_tau_def
        ssc_step_decay = self.ssc_step_decay_def
        ssc_step_decay_im = self.ssc_step_decay_im_def
        init_sal_discs = self.init_sal_discs_def
        init_sal_white = self.init_sal_white_def
        sal_image_max = self.sal_image_max_def
        retention_factor = self.retention_factor_def
        fixation_cycle = self.fixation_cycle_def
        sigma_lr = self.sigma_lr_def

        try:
            temperature = x[0]
            pred_tau = x[1]
            ssc_step_decay = x[2]
            ssc_step_decay_im = x[3]
            init_sal_discs = x[4]
            init_sal_white = x[5]
            sal_image_max = x[6]
            retention_factor = x[7]
            fixation_cycle = x[8]
            sigma_lr = x[9]
        except:
            pass
        '''
        temperature = self.values_all[0]
        pred_tau = self.values_all[1]
        ssc_step_decay = self.values_all[2]
        ssc_step_decay_im = self.values_all[3]
        init_sal_discs = self.values_all[4]
        init_sal_white = self.values_all[5]
        sal_image_max = self.values_all[6]
        retention_factor = self.values_all[7]
        fixation_cycle = self.values_all[8]
        sigma_lr = self.values_all[9]

        filename = '{}{}.par'.format(self.params_folder, self.rank)
        if self.verbose and self.rank==1:
            print 'worker {} opening file {} for writing.'.format(self.rank, filename)
        if not os.path.exists(self.params_folder):
            print 'Error: {} does not exist!'.format(self.params_folder)
            self.error = True
            return
        try:
            outputfile = open(filename, 'w')
        except IOError:
            print 'Error while opening file {}!'.format(filename)
            self.error = True
            return
        outputfile.write('FAILURE_RATE0:\t{}\n'.format(self.failure_rates[0]))
        outputfile.write('FAILURE_RATE1:\t{}\n'.format(self.failure_rates[1]))
        outputfile.write('FAILURE_RATE2:\t{}\n'.format(self.failure_rates[2]))
        outputfile.write('TEMPERATURE:\t{}\n'.format(temperature))
        outputfile.write('PRED_TAU:\t{}\n'.format(pred_tau))
        outputfile.write('SSC_STEP_DECAY:\t{}\n'.format(ssc_step_decay))
        outputfile.write('SSC_STEP_DECAY_IM:\t{}\n'.format(ssc_step_decay_im))
        outputfile.write('INIT_SAL_DISCS:\t{}\n'.format(init_sal_discs))
        outputfile.write('INIT_SAL_WHITE:\t{}\n'.format(init_sal_white))
        outputfile.write('SAL_IMAGE_MAX:\t{}\n'.format(sal_image_max))
        outputfile.write('RETENTION_FACTOR:\t{}\n'.format(retention_factor))
        outputfile.write('FIXATION_CYCLE:\t{}'.format(fixation_cycle))
        outputfile.write('SIGMA_LR:\t{}'.format(sigma_lr))

        outputfile.close()
        if self.verbose and self.rank==1:
            print 'worker {} closed file after writing.'.format(self.rank)


    def test_objective(self, x):
        #x_target = np.arange(1./(self.N_dim+1), 1., 1./(self.N_dim+1))
        x_target = np.array(self.default_values) + np.random.randn(self.N_dim)
        deviation = x_target - x
        #value = 100.0 * np.sqrt(np.dot(10.0*deviation, 0.1*deviation))
                                    # arbitrary objective function,
                                    #  minimal if x == xtarget
        value = np.linalg.norm(deviation)
        rates = np.zeros([3,2])
        return value, rates


    def objective(self, x):            # objective function
        if self.verbose and self.rank==1:
            print 'worker', self.rank, 'entering objective method'

        self.write_params()
        if self.error:
            return 0.0, np.zeros([3,2]), 1
        #command = './model2 {} {} {} > ./logs/{}_log.txt'.format(self.rank, self.outputpath, self.group, self.rank)
        command = './model {} {} {} > ./{}logs/{}_log.txt'.format(self.rank, self.outputpath, self.group, self.outputpath, self.rank)
            #'>', './{}logs/{}_log.txt'.format(self.outputpath, self.rank)]
        #command = ['./model', str(self.rank), self.outputpath, self.group]
        if self.verbose and self.rank==1:
	        print 'executing', command
        os.system(command)
        #pipe_file = open('./{}logs/{}_log.txt'.format(self.outputpath, self.rank), 'w')
        #subprocess.call(command, stdout=pipe_file, stderr=subprocess.STDOUT)
        #pipe_file.close()
        if self.verbose and self.rank==1:
            print 'model computation complete'

        rates, corrupt = rate_analysis.extract_mean_rates(self.rank, self.outputpath, self.verbose)
        if corrupt:
            N_corrupt = 1
        else:
            N_corrupt = 0
        diffs = [0,0,0]
        for i_rate in xrange(3):
            diffs[i_rate] = rates[i_rate][0] - rates[i_rate][1]
        #fix_ratios, F_disc_ratios, corrupt = ratio_analysis.extract_mean_ratios(self.rank, self.outputpath, self.verbose)
        #if corrupt:
        #    N_corrupt = 1
        #else:
        #    N_corrupt = 0

        value = 0.0
        if self.fitting_mode == 'rates':
            try:
              for i_session in xrange(3):
    #            if self.verbose and self.rank==1:
    #                print 'target_rates[{}]:'.format(i_session), self.target_rates[i_session]
                if not (self.target_rates[i_session] < -4000.0).all():
                  diff_rates = self.target_rates[i_session] - rates[i_session]
                  value += np.linalg.norm(diff_rates)
                  if self.verbose and self.rank==1:
                      print 'target_rates[{}] - rates[{}] = {} - {} = {}'.format(i_session, i_session,
                        self.target_rates[i_session], rates[i_session], diff_rates)
                      print 'fitness:', value
            except:
                print 'Error while evaluating rates!'
                self.error = True
        elif self.fitting_mode == 'diff':
            try:
              for i_session in xrange(3):
    #            if self.verbose and self.rank==1:
    #                print 'target_rates[{}]:'.format(i_session), self.target_rates[i_session]
                if not (self.target_diffs[i_session] < -4000.0).all():
                  diff_diffs = self.target_diffs[i_session] - diffs[i_session]
                  value += diff_diffs**2
                  if self.verbose and self.rank==1:
                      print 'target_diffs[{}] - diffs[{}] = {} - {} = {}'.format(i_session, i_session,
                        self.target_diffs[i_session], diffs[i_session], diff_diffs)
                      print 'fitness:', value
            except:
                print 'Error while evaluating diffs!'
                self.error = True
        else:
            print 'Error: fitting mode {} not recognized while evaluating samples!'.format(self.fitting_mode)
            self.error = True

        if self.verbose and self.rank==1:
            print 'PRED_TAU:', x[0]
            print 'TEMPERATURE: 0.0'#, x[1]
            print 'rates:', rates
            print 'deviation:', value

        return value, rates, diffs, N_corrupt




    def objective2(self, x):          # objective function using model2
        if self.verbose and self.rank==1:
            print 'worker', self.rank, 'entering objective method'

        rates = np.ones([3,2]) * -42.

        # actions: [white, funct, image, nonfunct]

        temperature = self.values_all[0]
        pred_tau = self.values_all[1]
        ssc_step_decay = self.values_all[2]
        ssc_step_decay_im = self.values_all[3]
        init_sal_discs = self.values_all[4]
        init_sal_white = self.values_all[5]
        sal_image_max = self.values_all[6]
        retention_factor = self.values_all[7]

        initial_int_sal = np.array([init_sal_white, init_sal_discs, 
            init_sal_white, init_sal_discs])
        hab_slowness = np.array([ssc_step_decay, ssc_step_decay, ssc_step_decay_im,
            ssc_step_decay])
        response_probs = np.zeros([4])
        if self.failure_rates[0] >= 0.0:    # failure_rate=-1 indicates filtered out session
            response_probs[1] = 1.0 - self.failure_rates[0]
        else:
            response_probs[1] = 1.0
        triggered_int_sal = np.array([-1.,-1.,sal_image_max,-1.])
        T_max = 600     # corresponds to 5 minutes if fixations happen every 0.5 seconds

        myAgent = Agent(initial_int_sal=initial_int_sal, T_max=T_max, 
            exploration_rate=temperature, learning_speed=pred_tau, 
            hab_slowness=hab_slowness, response_probs=response_probs,
            triggered_int_sal=triggered_int_sal, output=False, verbose=False)

        # experiment at T0
        data = myAgent.run()

        int_sal, probs, rates[0] = analyze_model2.analysis(data)

        # experiment at T1
        myAgent.int_sal = int_sal
        myAgent.probs = probs
        if self.failure_rates[1] >= 0.0:
            myAgent.response_probs[1] = 1.0 - self.failure_rates[1]
        else:
            myAgent.response_probs[1] = 1.0

        data = myAgent.run()

        int_sal, probs, rates[1] = analyze_model2.analysis(data)

        # experiment at T2
        myAgent.int_sal = initial_int_sal           # dishabituation
        myAgent.probs = probs * retention_factor    # forgetting
        if self.failure_rates[2] >= 0.0:
            myAgent.response_probs[1] = 1.0 - self.failure_rates[2]
        else:
            myAgent.response_probs[1] = 1.0

        data = myAgent.run()

        int_sal, probs, rates[2] = analyze_model2.analysis(data)

        if self.verbose and self.rank==1:
            print 'model computation complete'
        del myAgent
        
        # fitness calculation
        diffs = [0,0,0]
        for i_rate in xrange(3):
            diffs[i_rate] = rates[i_rate][0] - rates[i_rate][1]

        value = 0.0
        if self.fitting_mode == 'rates':
            try:
              for i_session in xrange(3):
    #            if self.verbose and self.rank==1:
    #                print 'target_rates[{}]:'.format(i_session), self.target_rates[i_session]
                if not (self.target_rates[i_session] < -4000.0).all():
                  diff_rates = self.target_rates[i_session] - rates[i_session]
                  value += np.linalg.norm(diff_rates)
                  if self.verbose and self.rank==1:
                      print 'target_rates[{}] - rates[{}] = {} - {} = {}'.format(i_session, i_session,
                        self.target_rates[i_session], rates[i_session], diff_rates)
                      print 'fitness:', value
            except:
                print 'Error while evaluating rates!'
                self.error = True
        elif self.fitting_mode == 'diff':
            try:
              for i_session in xrange(3):
    #            if self.verbose and self.rank==1:
    #                print 'target_rates[{}]:'.format(i_session), self.target_rates[i_session]
                if not (self.target_diffs[i_session] < -4000.0).all():
                  diff_diffs = self.target_diffs[i_session] - diffs[i_session]
                  value += diff_diffs**2
                  if self.verbose and self.rank==1:
                      print 'target_diffs[{}] - diffs[{}] = {} - {} = {}'.format(i_session, i_session,
                        self.target_diffs[i_session], diffs[i_session], diff_diffs)
                      print 'fitness:', value
            except:
                print 'Error while evaluating diffs!'
                self.error = True
        else:
            print 'Error: fitting mode {} not recognized while evaluating samples!'.format(self.fitting_mode)
            self.error = True

        if self.verbose and self.rank==1:
            print 'rates:', rates
            print 'deviation:', value

        N_corrupt = 0

        return value, rates, diffs, N_corrupt




    def sampling(self, x_mean, sigma, B_D):
        if self.verbose and self.rank==1:
            print 'worker', self.rank, 'entering sampling method with\nx_mean={}\nsigma={}\nB_D={}'.format(x_mean, sigma, B_D)
        while True:
            #resampling = False
            z = np.random.randn(self.N_dim) # standard normally distributed vector
            x = x_mean + sigma * (np.dot(B_D, z))
                                    # add mutation, Eq. 37
            if 'temperature' in self.value_names_fit:
                i_name = self.value_names_fit.index('temperature')
                if x[i_name] <= 0.0:
                    continue
            if 'pred_tau' in self.value_names_fit:
                i_name = self.value_names_fit.index('pred_tau')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
            if 'ssc_step_decay' in self.value_names_fit:
                i_name = self.value_names_fit.index('ssc_step_decay')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
            if 'ssc_step_decay_im' in self.value_names_fit:
                i_name = self.value_names_fit.index('ssc_step_decay_im')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
                if 'ssc_step_decay' in self.value_names_fit:
                    i_dec = self.value_names_fit.index('ssc_step_decay')
                    if x[i_name] < x[i_dec]:
                        continue
                else:
                    if x[i_name] < self.ssc_step_decay_def:
                        continue
            if 'init_sal_discs' in self.value_names_fit:
                i_name = self.value_names_fit.index('init_sal_discs')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
                if 'init_sal_white' in self.value_names_fit:
                    i_white = self.value_names_fit.index('init_sal_white')
                    if x[i_white] >= x[i_name]:
                        continue
                else:
                    if self.init_sal_white_def >= x[i_name]:
                        continue
                if 'sal_image_max' in self.value_names_fit:
                    i_im = self.value_names_fit.index('sal_image_max')
                    if x[i_name] >= x[i_im]:
                        continue
                else:
                    if x[i_name] >= self.sal_image_max_def:
                        continue
            if 'init_sal_white' in self.value_names_fit:
                i_name = self.value_names_fit.index('init_sal_white')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
                if 'init_sal_discs' not in self.value_names_fit:
                    if x[i_name] >= self.init_sal_discs_def:
                        continue
                if 'sal_image_max' not in self.value_names_fit:
                    if x[i_name] >= self.sal_image_max_def:
                        continue
            if 'sal_image_max' in self.value_names_fit:
                i_name = self.value_names_fit.index('sal_image_max')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
                if 'init_sal_discs' not in self.value_names_fit:
                    if x[i_name] <= self.init_sal_discs_def:
                        continue
                if 'init_sal_white' not in self.value_names_fit:
                    if x[i_name] <= self.init_sal_white_def:    
                        continue
            if 'retention_factor' in self.value_names_fit:
                i_name = self.value_names_fit.index('retention_factor')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
            if 'fixation_cycle' in self.value_names_fit:
                i_name = self.value_names_fit.index('fixation_cycle')
                if x[i_name] <= 0.0 or x[i_name] >= 1.0:
                    continue
            if 'sigma_lr' in self.value_names_fit:
                i_name = self.value_names_fit.index('sigma_lr')
                if x[i_name] <= 0.0:
                    continue
            break

            '''
            if x[0]>0:
                if self.N_dim > 1:
                    if ((x[1:] > 0).all() and (x[1:9] < 1).all()):   # parameters within ]0,1[
                        if self.N_dim > 6:
                            if x[6]>x[4]>x[5]:      # image sal > disc sal > background sal
                                resampling = False
                        else:
                            resampling = False
                else:
                    resampling = False
            if self.verbose and self.rank==1 and resampling:
                print 'Sampled invalid parameters x =', x
            '''
        return x, z


    def terminate_process(self):
        print 'Master sending kill signal to workers'
        self.terminate = True
        for i_worker in xrange(1,self.n_workers):
            self.comm.send((None, None, None, True, self.error), dest=i_worker, tag=i_worker)
        return


    def plot_fit(self):
        target_rates = self.target_rates.T
        rates = self.rates[0].copy()
        rates = rates.T

        plt.figure()
        ax = plt.subplot()
        plt.title('Mean observed pattern rates')
        plt.plot(target_rates[0], 'bo')
        plt.plot(target_rates[1], 'ro')
        plt.plot(rates[0], 'bx')
        plt.plot(rates[1], 'rx')
        plt.xlim(-0.5, 2.5)
        plt.xticks(range(3), ['T0', 'T1', 'T2'])
        plt.xlabel('Test times')
        plt.ylabel('Patterns per minute')
        plt.savefig(self.outputpath+'fit.jpg')

        plt.figure()
        ax = plt.subplot()
        plt.title('Rate deviation')
        plt.plot(self.fitness_all, 'k-')
        plt.xlabel('Time steps')
        plt.savefig(self.outputpath+'deviation.jpg')

        plt.figure()
        ax = plt.subplot()
        plt.title('Sigma')
        plt.plot(self.sigma_all, 'k-')
        plt.xlabel('Time steps')
        plt.savefig(self.outputpath+'sigma.jpg')

        if self.verbose:
            print 'raw rates_all:', self.rates_all
        rates_all = np.array(self.rates_all)
        rates_all = rates_all.transpose((1,2,0))
        if self.verbose:
            print 'converted rates_all:', rates_all
            print 'target_rates:', self.target_rates
        n_steps = len(self.rates_all)
        for i_session in xrange(3):
         target_funct = self.target_rates[i_session][0] * np.ones(n_steps)
         target_nonfunct = self.target_rates[i_session][1] * np.ones(n_steps)

         plt.figure()
         ax = plt.subplot()
         plt.title('Rates of session {}'.format(i_session))
         plt.plot(rates_all[i_session][0], 'b-')
         plt.plot(rates_all[i_session][1], 'r-')
         if (target_funct > -100.0).all():
             plt.plot(target_funct, 'b:')
             plt.plot(target_nonfunct, 'r:')
         plt.xlabel('Time steps')
         plt.savefig(self.outputpath+'rates_{}.jpg'.format(i_session))

        if self.verbose:
            print 'raw x_all:', self.x_all
        x_all = np.array(self.x_all)
        x_all = x_all.T
        if self.verbose:
            print 'converted x_all:', x_all

        for i_x in xrange(self.N_dim):
            value_name = self.value_names_fit[i_x]
            plt.figure()
            plt.title(value_name)
            plt.plot(x_all[i_x], 'k-')
            plt.xlabel('Time steps')
            plt.savefig(self.outputpath+value_name+'.jpg')

        plt.close('all')


    def store_params(self, i_max):
        filename = '{}{}.par'.format(self.params_folder, i_max)
        #os.system('rm {}*.par'.format(self.outputpath))
        #os.system('cp {} {}'.format(filename, self.outputpath))
        try:
            shutil.copy(filename, self.outputpath)
        except IOError:
            print 'Error while copying file {} to {}!'.format(filename, self.outputpath)
            self.error = True


    def store_solutions(self):
        if self.verbose:
            print 'storing solutions:', self.solutions

        filename_obj = '{}solutions.obj'.format(self.outputpath)
        #os.system('rm {}'.format(filename_obj))
        try:
            os.remove(filename_obj)
        except:
            pass
        solutions = self.solutions
        try:
            solutions.sort(key=aux.sort_key_funct)
        except:
            print 'Warning: exception during sorting of solutions!\nsolutions:', solutions
        file_obj = open(filename_obj, 'wb')
        cPickle.dump(solutions, file_obj)
        file_obj.close()

        filename_xls = '{}solutions.xls'.format(self.outputpath)
        try:
            os.remove(filename_xls)
        except:
            pass
        #os.system('rm {}'.format(filename_xls))
        file_xls = open(filename_xls, 'w')
        if self.fitting_mode == 'rates':
            file_xls.write('fitness\tfunct1\tnonfunct1\tfunct2\tnonfunct2\tfunct3\tnonfunct3')
        elif self.fitting_mode == 'diff':
            file_xls.write('fitness\tdiff1\tdiff2\tdiff3')
        for value_name in self.value_names_fit:
            file_xls.write('\t{}'.format(value_name))
        file_xls.write('\ntarget')
        if self.fitting_mode == 'rates':
            for i_session in xrange(3):
                file_xls.write('\t{}\t{}'.format(self.target_rates[i_session][0], self.target_rates[i_session][1]))
        elif self.fitting_mode == 'diff':
            for i_session in xrange(3):
                file_xls.write('\t{}'.format(self.target_diffs[i_session]))
        if self.verbose:
            print 'iterating over solutions:', solutions
        for line in solutions:
            if self.verbose:
                print 'solution line:', line
            file_xls.write('\n{}'.format(line[0]))
            if self.fitting_mode == 'rates':
                for i_session in xrange(3):
                    file_xls.write('\t{}\t{}'.format(line[2][i_session][0], line[2][i_session][1]))
            elif self.fitting_mode == 'diff':
                for i_session in xrange(3):
                    file_xls.write('\t{}'.format(line[1][i_session]))
            for value in line[3]:
                file_xls.write('\t{}'.format(value))
        file_xls.close()


    def run(self):
      #try:
        #print 'worker {} entered run method'.format(self.rank)
        if self.rank > 0:
            while True:
                if self.verbose and self.rank==1:
                    print 'worker', self.rank, 'waiting for params'
                x_mean, sigma, B_D, terminate, error = self.comm.recv(source=0, tag=self.rank)
                if self.verbose and self.rank==1:
                    print 'worker', self.rank, 'received params'
                if error:
                    print 'worker {} shutting down with error signal (name={})'.format(self.rank, self.name)
                    return True
                if terminate:
                    if self.verbose:
                        print 'worker {} shutting down (name={})'.format(self.rank, self.name)
                    return False

                x, z = self.sampling(x_mean, sigma, B_D)
                for i_x in xrange(self.N_dim):
                    x_i = x[i_x]
                    name_i = self.value_names_fit[i_x]
                    i_value = self.value_names_fit_dic[name_i]
                    self.values_all[i_value] = x_i
                #fitness, rates = self.test_objective(x)
                if self.model == 1:
                    fitness, rates, diffs, N_corrupt = self.objective(x) 
                elif self.model == 2:
                    fitness, rates, diffs, N_corrupt = self.objective2(x)
                rates = np.array(rates)
                diffs = np.array(diffs)
                if self.memory_profiler:
                    memory = self.memory_profiler.heap().size
                else:
                    memory = None
                if self.verbose and self.rank==1:
                    print 'worker {} sending values:\nx={}\nz={}\nfitness={}\ndiffs={}\nrates={}\nN_corrupt={}\nmemory={}\nerror={}'.format(self.rank,
                        x, z, fitness, diffs, rates, N_corrupt, memory, self.error)
                self.comm.send((x, z, fitness, rates, diffs, N_corrupt, memory, self.error), dest=0, tag=self.rank)
                if self.verbose and self.rank==1:
                    print 'worker', self.rank, 'sent values'

        else:

        #######################################################
        # Master Generation Loop
        #######################################################

            self.x_all = []
            self.fitness_all = []
            self.rates_all = []
            self.diffs_all = []
            self.sigma_all = []
            self.i_reset = 0
            self.solutions = []
            self.sigma0 = 0.1
            self.N_corrupt = 0

            i_count = 0                     # the next 40 lines contain the 20 lines of interesting code
            while (self.i_reset < self.max_reset) and not self.terminate:

                if self.n_workers == 1:
                    if self.verbose:
                        print 'single worker mode generation'
                    # Generate and evaluate lambda offspring                    
                    for k in xrange(self.lambda_):
                        self.x[k] , self.z[k]= self.sampling(self.x_mean, self.sigma, self.B_D)
            
                        self.fitness[k], self.rates[k], self.diffs[k], self.corrupts[k] = self.objective(self.x[k])
                                                # objective function call
                        i_count += 1
                    
                else:
                    for rank in xrange(1,self.n_workers):
                        if self.verbose and rank==1:
                            print 'master sending params to workers'
                        self.comm.send((self.x_mean, self.sigma, self.B_D, self.terminate, self.error), dest=rank, tag=rank)
                    if self.memory_profiler:
                        self.memory_all[0] = self.memory_profiler.heap().size
                    for rank in xrange(1,self.n_workers):
                        if self.verbose and rank==1:    
                            print 'master awaiting values from workers'
                        (self.x[rank-1], self.z[rank-1], self.fitness[rank-1], self.rates[rank-1], self.diffs[rank-1],
                            self.corrupts[rank-1], 
                            self.memory_all[rank], self.errors_all[rank-1]) = self.comm.recv(source=rank, tag=rank)
                            # memory_all[0] is memory of master
                        if self.errors_all.any():
                            self.terminate = True
                            self.error = True
                            print 'Kill signal received. Shutting down. (name={})'.format(self.name)
                            self.terminate_process()
                            return None, None, True
                        if self.verbose and rank==1:
                            print 'master received values:', self.x[rank-1], self.fitness[rank-1], 'from worker', rank
                        i_count += 1
                    #print 'i_count:', i_count

                self.N_corrupt += self.corrupts.sum()

                if self.verbose:
                    print 'x:', self.x
                    print 'fitness:', self.fitness
                    print 'corrupts:', self.corrupts

                # Sort by fitness and compute weighted mean into x_mean
                indices = np.arange(self.lambda_)
                to_sort = zip(self.fitness, self.diffs.tolist(), self.rates.tolist(), indices)
                                            # minimization
                if self.verbose:
                    print 'to_sort:', to_sort
                to_sort.sort()
                self.fitness, self.diffs, self.rates, indices = zip(*to_sort)
                self.fitness = np.array(self.fitness)
                self.rates = np.array(self.rates)
                self.diffs = np.array(self.diffs)
                indices = np.array(indices)
                index_max = indices[0] + 1  # offset accounts for master rank 0

                if self.model==1:
                    self.store_params(index_max)

                self.x_mean = np.zeros(self.N_dim)
                self.z_mean = np.zeros(self.N_dim)
                fitness_mean = 0.0
                for i in xrange(self.mu):
                    self.x_mean += self.weights[i] * self.x[indices[i]]
                                            # recombination, Eq. 39
                    self.z_mean += self.weights[i] * self.z[indices[i]]
                                            # == D^-1 * B^T * (x_mean-x_old)/sigma
                    fitness_mean += self.weights[i] * self.fitness[indices[i]]
                
                # Cumulation: Update evolution paths
                self.p_s = (1.0-self.c_s)*self.p_s + (np.sqrt(self.c_s*(2.0-self.c_s)*self.mu_eff)) * np.dot(self.B,self.z_mean)
                                            # Eq. 40
                self.h_sig = int(np.linalg.norm(self.p_s) / np.sqrt(1.0-(1.0-self.c_s)**(2.0*i_count/self.lambda_))/self.chi_N < 1.4+2.0/(self.N_dim+1.0))
                self.p_c = (1.0-self.c_c)*self.p_c + self.h_sig * np.sqrt(self.c_c*(2.0-self.c_c)*self.mu_eff) * np.dot(self.B_D,self.z_mean)
                                            # Eq. 42

                # Adapt covariance matrix C
                if self.verbose:
                    print 'B:\n', self.B
                    print 'D:\n', self.D
                    print 'z[:mu]:\n', self.z[:self.mu]
                    print 'p_c:\n', self.p_c
                    print 'C before adaptation:\n', self.C
                    print 'np.diag(weights):\n', np.diag(self.weights)
                
                #self.C = (1.0-self.c_1-self.c_mu)*self.C + self.c_1*(np.dot(self.p_c,self.p_c.T) + (1.0-self.h_sig)*self.c_c*(2.0-self.c_c)*self.C) + self.c_mu*np.dot(np.dot((np.dot(self.B_D, self.z[indices[:self.mu]].T)),np.diag(self.weights)),(np.dot(self.B_D, self.z[indices[:self.mu]].T)).T)
                                            # regard old matrix plus rank one update plus minor correction plus rank mu update, Eq. 43
                C_1 = (1.0 - self.c_1 - self.c_mu) * self.C
                C_2 = self.c_1 * (np.dot(self.p_c, self.p_c.T) + (1.0 - self.h_sig) * self.c_c * (2.0 - self.c_c) * self.C)
                #C_3 = self.c_mu * np.dot(np.dot(np.dot(self.B_D, self.z[indices[:self.mu]].T), np.diag(self.weights)), np.dot(self.B_D, self.z[indices[:self.mu]].T).T)
                C_3_1 = np.dot(self.B_D, self.z[indices[:self.mu]].T)
                C_3_2 = np.dot(C_3_1, np.diag(self.weights))
                C_3_3 = np.dot(C_3_2, C_3_1.T)
                C_3 = self.c_mu * C_3_3
                self.C = C_1 + C_2 + C_3

                if self.verbose:
                    print 'C_1:\n', C_1
                    print 'C_2:\n', C_2
                    print 'C_3_1:\n', C_3_1
                    print 'C_3_2:\n', C_3_2
                    print 'C_3_3:\n', C_3_3
                    print 'C_3:\n', C_3
                    print 'C after adaptation:\n', self.C


                # Adapt step-size sigma
                self.sigma = self.sigma * np.exp((self.c_s/self.damps) * (np.linalg.norm(self.p_s)/self.chi_N - 1.0))
                                            # Eq. 41

                # Update B and D from C
                if i_count - self.i_eigen > self.lambda_/(self.c_1+self.c_mu)/self.N_dim/10.0:
                                            # to achieve O(N**2)
                    self.i_eigen = i_count
                    if self.verbose:            
                        print 'C before symmetry enforcing:\n', self.C
                    self.C = np.triu(self.C) + np.triu(self.C,1).T
                                            # enforce symmetry
                    if self.verbose:
                        print 'C after symmetry enforcing:\n', self.C
                    self.D, self.B = np.linalg.eig(self.C) # eigen decomposition, B==normalized eigenvectors?
                    if self.verbose:
                        print 'B:\n', self.B
                        print 'D before diagonalization:\n', self.D
                        print 'np.diag(D):\n', np.diag(self.D)
                    self.D = np.diag(np.sqrt(self.D))
                                            # D contains standard deviations now
                    if self.verbose:
                        print 'D after diagonalization:\n', self.D
                    self.B_D = np.dot(self.B, self.D)

                # Escape flat fitness, or better terminate?
                if self.fitness[0] == self.fitness[int(np.ceil(0.7*self.lambda_))-1]:
                    self.sigma *= np.exp(0.2+self.c_s/self.damps)
                    print 'warning: flat fitness, consider reformulating the objective'
                    #print 'int(np.ceil(0.7*self.lambda_))-1:', int(np.ceil(0.7*self.lambda_))-1
                    #print 'fitness:', self.fitness

                while len(self.x_recent) > self.convergence_interval - 1:
                    self.x_recent.popleft()
                while len(self.fitness_recent) > self.convergence_interval - 1:
                    self.fitness_recent.popleft()
                self.x_recent.append(self.x_mean)
                self.fitness_recent.append(fitness_mean)
                if self.verbose:
                    print 'x_recent:', self.x_recent
                    print 'fitness_recent:', self.fitness_recent

                if ((np.ptp(self.x_recent, axis=0) < self.ptp_stop).all()) and len(self.x_recent)==self.convergence_interval:
                    print 'parameters converged. reinitializing search.'
                    self.i_reset += 1
                    self.solutions.append([self.fitness[0], self.diffs[0], self.rates[0], self.x[0]])
                    self.store_solutions()
                    self.reinit()
                    self.sigma = self.sigma0
                    if self.sigma0 < 0.5:
                        self.sigma0 += 0.05
                elif (np.ptp(self.fitness_recent) < self.ptp_stop) and len(self.fitness_recent)==self.convergence_interval:
                    print 'fitness converged. reinitializing search.'
                    self.i_reset += 1
                    self.solutions.append([self.fitness[0], self.diffs[0], self.rates[0], self.x[0]])
                    self.store_solutions()
                    self.reinit()
                    self.sigma = self.sigma0
                    if self.sigma0 < 0.5:
                        self.sigma0 += 0.05
                if self.i_reset == self.max_reset:
                    print 'reinitialization limit reached. terminating search.'
                    self.terminate = True
                    self.terminate_process()
                    break

                if self.verbose:
                    print 'appending to x_all:', self.x[0]
                self.x_all.append(self.x[0].copy())
                if self.verbose:
                    print 'x_all:', self.x_all
                self.fitness_all.append(self.fitness[0])
                self.rates_all.append(self.rates[0])
                self.diffs_all.append(self.diffs[0])
                self.sigma_all.append(self.sigma)
                print '\n', i_count, ': deviation', self.fitness[0], '\nsigma:', self.sigma,\
                    '\ndiffs:', self.diffs[0],\
                    '\nrates:', self.rates[0], '\nparams:', self.x[0],\
                    '\nN_corrupt:', self.N_corrupt
                if self.verbose:
                    print 'solutions:', self.solutions
                self.plot_fit()
            
                if self.memory_profiler:
                    for i_worker in xrange(self.n_workers):
                        self.memory_profiling_file.write('{:.2f}\t'.format(self.memory_all[i_worker]/1024.**2))
                    self.memory_profiling_file.write('\n')
                    self.memory_profiling_file.flush()

                # Break if fitness is good enough
                if self.fitness[0] <= self.fitness_stop:
                    print 'fitness criterion reached (fitness={}, fitness_stop={})'.format(self.fitness[0], self.fitness_stop)
                    self.solutions.append([self.fitness[0], self.diffs[0], self.rates[0], self.x[0]])
                    self.store_solutions()
                    self.terminate = True
                    self.terminate_process()
                    break


            x_min = self.x[0]
            self.solutions.sort(key=aux.sort_key_funct)
            best_solution = self.solutions[0]

            print 'found solution:'
            for i in xrange(self.N_dim):
                print '{}: {}'.format(self.value_names_fit[i], x_min[i])
            #print 'true solution:', self.x_target

            return best_solution, index_max, self.value_names_fit, self.error
        '''
      except:
        print '\nError encountered in Optimization!'
        traceback.print_exc()
        print ''
        if self.rank == 0:
            self.terminate_process()
        self.terminate = True

        return None
        '''
