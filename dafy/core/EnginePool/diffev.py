'''File: diffev.py an implementation of the differential evolution algoithm
for fitting.
Programmed by: Matts Bjorck
Last changed: 2008 11 23
'''
try:
    from mpi4py import MPI#cautions:this line could be buggy when you have mpi4py installed on your computer
except:
    MPI_RUN=False
MPI_RUN=False
from numpy import *
import _thread as thread
import time
import psutil
import numpy as np
from scipy.optimize import curve_fit
import random as random_mod
import sys, os, pickle
from PyQt5 import QtCore
import logging
split_jobs = 1

if MPI_RUN:
    __parallel_loaded__ = True
    _cpu_count = size
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
    comm_group= comm.Split(color=rank/(size/split_jobs),key= rank)
    size_group= comm_group.Get_size()
    rank_group= comm_group.Get_rank()
else:
    __parallel_loaded__ = False
    _cpu_count = 1

if not MPI_RUN:
    try:
        import ray
        _cpu_count = psutil.cpu_count(logical=False) 
        if not ray.is_initialized():
            ray.init(num_cpus=_cpu_count)
            __parallel_loaded__ = True
            logging.root.info('ray is loaded success')
        else:
            logging.root.info('ray is already initialized!')
    except:
        pass
#__parallel_loaded__ = False
# import model

from dafy.core.EnginePool.Simplex import Simplex

# Add current path to the system paths
# just in case some user make a directory change
sys.path.append(os.getcwd())

# class: DiffEv
class DiffEv:
    '''
    Class DiffEv
    Contains the implemenetation of the differential evolution algorithm.
    It also contains thread support which is activated by the start_fit
    function.
    '''
    def __init__(self,split_jobs=1):

        # Mutation schemes implemented
        self.mutation_schemes = [self.best_1_bin, self.rand_1_bin,\
            self.best_either_or, self.rand_either_or, self.jade_best, self.simplex_best_1_bin]
        try:
            self.model = model.Model()
        except:
            pass
        self.split_jobs=split_jobs
        self.km = 0.7 # Mutation constant
        self.kr = 0.7 # Cross over constant
        self.pf = 0.5 # probablility for mutation
        self.c = 0.07
        self.simplex_interval = 5 #Interval of running the simplex opt
        self.simplex_step = 0.05 #first step as a fraction of pop size
        self.simplex_n = 0.0 # Number of individuals that will be optimized by simplex
        self.simplex_rel_epsilon = 1000 # The relative epsilon - convergence critera
        self.simplex_max_iter = 100 # THe maximum number of simplex runs
        # Flag to choose beween the two alternatives below
        self.use_pop_mult = False
        self.pop_mult = 3 # Set the pop_size to pop_mult * # free parameters
        self.pop_size = 10 # Set the pop_size only

        # Flag to choose between the two alternatives below
        self.use_max_generations = True
        self.max_generations = 500 # Use a fixed # of iterations
        self.max_generation_mult = 6 # A mult const for max number of iter

        # Flag to choose whether or not to use a starting guess
        self.use_start_guess = False
        # Flag to choose wheter or not to use the boundaries
        self.use_boundaries = True

        # Sleeping time for every generation
        self.sleep_time = 0.2
        # Allowed disagreement between the two different fom
        # evaluations
        self.fom_allowed_dis = 1e-10
        # Flag if we should use parallel processing
        self.use_parallel_processing = __parallel_loaded__
        if __parallel_loaded__:
            self.processes = _cpu_count
        else:
            self.processes = 0
        self.chunksize = 1
        # Flag for using autosave
        self.use_autosave = True
        # autosave interval in generations
        self.autosave_interval = 100

        # Functions that are user definable
        self.plot_output = default_plot_output
        self.text_output = default_text_output
        self.parameter_output = default_parameter_output
        self.autosave = defualt_autosave
        self.fitting_ended = default_fitting_ended


        # Definition for the create_trial function
        self.create_trial = self.best_1_bin
        self.update_pop = self.standard_update_pop
        self.init_new_generation = self.standard_init_new_generation

        # Control flags:
        self.running = False # true if optimization is running
        self.stop = False # true if the optimization should stop
        self.setup_ok = False # True if the optimization have been setup
        self.error = False # True/string if an error ahs occured

        # Logging variables
        # Maximum number of logged elements
        self.max_log = 100000
        self.fom_log = array([[0,0]])[0:0]
        #self.par_evals = array([[]])[0:0]

        self.par_evals = CircBuffer(self.max_log, buffer = array([[]])[0:0])
        #self.fom_evals = array([])
        self.fom_evals = CircBuffer(self.max_log)

    def safe_copy(self, object):
        '''safe_copy(self, object) --> None

        Does a safe copy of object to this object. Makes copies of everything
        if necessary. The two objects become decoupled.
        '''
        self.km = object.km # Mutation constant
        self.kr = object.kr # Cross over constant
        self.pf = object.pf # probablility for mutation

        # Flag to choose beween the two alternatives below
        self.use_pop_mult = object.use_pop_mult
        self.pop_mult = object.pop_mult
        self.pop_size = object.pop_size

        # Flag to choose between the two alternatives below
        self.use_max_generations = object.use_max_generations
        self.max_generations = object.max_generations
        self.max_generation_mult = object.max_generation_mult

        # Flag to choose whether or not to use a starting guess
        self.use_start_guess = object.use_start_guess
        # Flag to choose wheter or not to use the boundaries
        self.use_boundaries = object.use_boundaries

        # Sleeping time for every generation
        self.sleep_time = object.sleep_time
        # Flag if we should use parallel processing
        if __parallel_loaded__:
            self.use_parallel_processing = object.use_parallel_processing
        else:
            self.use_parallel_processing = False

        # Definition for the create_trial function
        #self.create_trial = object.create_trial

        # True if the optimization have been setup
        self.setup_ok = object.setup_ok

        # Logging variables
        self.fom_log = object.fom_log[:]
        self.par_evals.copy_from(object.par_evals)
        self.fom_evals.copy_from(object.fom_evals)

        if self.setup_ok:
            self.n_pop = object.n_pop
            self.max_gen = object.max_gen

            # Starting values setup
            self.pop_vec = object.pop_vec

            self.start_guess = object.start_guess

            self.trial_vec = object.trial_vec
            self.best_vec = object.best_vec

            self.fom_vec = object.fom_vec
            self.best_fom = object.best_fom
            # Not all implementaions has these copied within their files
            # Just ignore if an error occur
            try:
                self.n_dim = object.n_dim
                self.par_min = object.par_min
                self.par_max = object.par_max
            except:
                pass

    def pickle_string(self, clear_evals = False):
        '''Pickle the object.

        Saves a copy into a pickled string note that the dynamic
        functions will not be saved. For normal use this is taken care of
        outside this class with the config object.
        '''
        cpy = DiffEv()
        cpy.safe_copy(self)
        if clear_evals:
            cpy.par_evals.buffer = cpy.par_evals.buffer[0:0]
            cpy.fom_evals.buffer = cpy.fom_evals.buffer[0:0]
        cpy.create_trial = None
        cpy.update_pop = None
        cpy.init_new_generation = None
        cpy.plot_output = None
        cpy.text_output = None
        cpy.parameter_output = None
        cpy.autosaves = None
        cpy.fitting_ended = None
        cpy.model = None
        cpy.mutation_schemes = None

        return pickle.dumps(cpy)

    def pickle_load(self, pickled_string):
        '''load_pickles(self, pickled_string) --> None

        Loads the pickled string into the this object. See pickle_string.
        '''
        self.safe_copy(pickle.loads(pickled_string,encoding='latin1'))


    def reset(self):
        ''' reset(self) --> None

        Resets the optimizer. Note this has to be run if the optimizer is to
        be restarted.
        '''
        self.setup_ok = False

    def connect_model(self, model):
        '''connect_model(self, model) --> None

        Connects the model [model] to this object. Retrives the function
        that sets the variables  and stores a reference to the model.
        '''
        # Retrive parameters from the model
        (par_funcs, start_guess, par_min, par_max, par_funcs_link) = model.get_fit_pars()

        # Control parameter setup
        self.par_min = array(par_min)
        self.par_max = array(par_max)
        self.par_funcs = par_funcs
        self.par_funcs_link = par_funcs_link
        self.model = model
        self.n_dim = len(par_funcs)
        if not self.setup_ok:
            self.start_guess = start_guess

    def init_fitting(self, model):
        '''
        Function to run before a new fit is started with start_fit.
        It initilaize the population and sets the limits on the number
        of generation and the population size.
        '''
        self.connect_model(model)
        if self.use_pop_mult:
            self.n_pop = int(self.pop_mult*self.n_dim)
        else:
            self.n_pop = int(self.pop_size)
        if self.use_max_generations:
            self.max_gen = int(self.max_generations)
        else:
            self.max_gen = int(self.max_generation_mult*self.n_dim*self.n_pop)
        if not MPI_RUN:
            # Starting values setup
            self.pop_vec = [self.par_min + random.rand(self.n_dim)*(self.par_max -\
             self.par_min) for i in range(self.n_pop)]

            if self.use_start_guess:
                self.pop_vec[0] = array(self.start_guess)

            self.trial_vec = [zeros(self.n_dim) for i in range(self.n_pop)]
            self.best_vec = self.pop_vec[0]

        self.fom_vec = zeros(self.n_dim)
        self.best_fom = 1e20

        # Storage area for JADE archives
        self.km_vec = ones(self.n_dim)*self.km
        self.kr_vec = ones(self.n_dim)*self.kr


        # Logging varaibles
        self.fom_log = array([[0,1]])[0:0]
        self.par_evals = CircBuffer(self.max_log,
                                    buffer = array([self.par_min])[0:0])
        #self.fom_evals = array([])
        self.fom_evals = CircBuffer(self.max_log)
        # Number of FOM evaluations
        self.n_fom = 0
        #self.par_evals.reset(array([self.par_min])[0:0])
        #self.fom_evals.reset()

        if MPI_RUN:
            if rank==0:self.text_output('DE initilized')
        else:
            self.text_output('DE initilized')

        # Remeber that everything has been setup ok
        self.setup_ok = True

    def init_fom_eval(self):
        '''init_fom_eval(self) --> None

        Makes the eval_fom function
        '''
        # MPI_RUN = False
        # self.use_parallel_processing = True
        # self.__parallel_loaded__ = True
        if not MPI_RUN:
            # Setting up for parallel processing
            if self.use_parallel_processing and __parallel_loaded__:
                self.text_output('Setting up a pool of workers ...')
                print('Setting up a pool of workers ...')
                self.setup_parallel()
                self.eval_fom = self.calc_trial_fom_parallel
            else:
                self.eval_fom = self.calc_trial_fom
        else:
            if __parallel_loaded__:
                self.setup_parallel_mpi()
                self.eval_fom = self.calc_trial_fom_parallel_mpi
            else:
                self.eval_fom = self.calc_trial_fom

    def start_fit(self,signal = None, signal_fitended = None):
        '''
        Starts fitting in a seperate thred.
        '''
        # If it is not already running
        if not self.running:
            #Initilize the parameters to fit
            self.reset()
            self.init_fitting(self.model)
            self.init_fom_eval()
            self.stop = False
            # Start fitting in a new thread
            # thread.start_new_thread(self.optimize, ())
            self.text_output('Starting the fit...')
            logging.root.info('Starting the fit ...')
            print('Starting the fit...')
            self.optimize(signal = signal, signal_fitended = signal_fitended)
            # self.optimize()
            # For debugging
            #self.optimize()
            #self.running = True
            return True
        else:
            self.text_output('Fit is already running, stop and then start')
            logging.root.info('Fit is already running, stop and then start')
            return False

    def stop_fit(self):
        '''
        Stops the fit if it has been started in a seperate theres
        by start_fit.
        '''
        # If not running stop
        if self.running:
            self.stop = True
            self.text_output('Trying to stop the fit...')
        else:
            self.text_output('The fit is not running')

    def resume_fit(self, model):
        '''
        Resumes the fitting if has been stopped with stop_fit.
        '''
        if not self.running:
            self.stop = False
            self.connect_model(model)
            self.init_fom_eval()
            n_dim_old = self.n_dim
            if self.n_dim == n_dim_old:
                thread.start_new_thread(self.optimize, ())
                self.text_output('Restarting the fit...')
                self.running = True
                return True
            else:
                self.text_output('The number of parameters has changed'\
                ' restart the fit.')
                return False
        else:
            self.text_output('Fit is already running, stop and then start')
            return False

    def optimize_partial(self):
        '''
        Method implementing the main loop of the differential evolution
        algorithm. Note that this method does not run in a separate thread.
        For threading use start_fit, stop_fit and resume_fit instead.
        '''

        self.text_output('Calculating start FOM ...')
        self.running = True
        self.error = False
        self.n_fom = 0
        #print self.pop_vec
        #eval_fom()
        #self.fom_vec = self.trial_fom[:]
        # Old leftovers before going parallel
        self.fom_vec = [self.calc_fom(vec) for vec in self.pop_vec]
        [self.par_evals.append(vec, axis = 0)\
                    for vec in self.pop_vec]
        [self.fom_evals.append(vec) for vec in self.fom_vec]
        #print self.fom_vec
        best_index = argmin(self.fom_vec)
        #print self.fom_vec
        #print best_index
        self.best_vec = copy(self.pop_vec[best_index])
        #print self.best_vec
        self.best_fom = self.fom_vec[best_index]
        #print self.best_fom
        if len(self.fom_log) == 0:
            self.fom_log = r_[self.fom_log,\
                                [[len(self.fom_log),self.best_fom]]]
        # Flag to keep track if there has been any improvemnts
        # in the fit - used for updates
        self.new_best = True

        self.text_output('Going into optimization ...')

        # Update the plot data for any gui or other output
        #self.plot_output(self)

        #self.parameter_output(self)

    def return_covariance_matrix(self, fom_level = 0.15):
        import pandas as pd
        if len(self.par_evals)==0:
            return pd.DataFrame(np.identity(2))
        condition = (self.fom_evals.array()+1)<(self.model.fom+1)*(1+fom_level)
        target_matrix = self.par_evals[condition]
        df = pd.DataFrame(target_matrix)
        corr = df.corr()
        corr.index += 1
        corr = corr.rename(columns = lambda x:str(int(x)+1))
        return corr

    def return_sensitivity(self, max_epoch = 200, epoch_step = 0.1):
        import copy
        index_fit_pars = [i for i in range(len(self.model.parameters.data)) if self.model.parameters.data[i][2]]
        #par_names = ['{}.'.format(i) for i in range(1,len(index_fit_pars)+1)]
        #print(par_names)
        epoch_list = [0]*len(index_fit_pars)
        fom_diff_list = [0]*len(index_fit_pars)
        for i in index_fit_pars:
            par = self.model.parameters.get_value(i, 0)
            print('Screen par {} now!'.format(par))
            current_value = self.model.parameters.get_value(i, 1)
            current_fom = self.model.fom
            current_vec = copy.deepcopy(self.best_vec)
            epoch = 0
            while epoch<max_epoch:
                epoch = epoch + 1
                #self.model.parameters.set_value(i, 1, current_value*(1+epoch_step*epoch))
                #self.model.simulate()
                current_vec[index_fit_pars.index(i)] = current_value+abs(current_value)*epoch_step*epoch
                fom = self.calc_fom(current_vec)
                #offset off 1 is used just in case the best fom is very close to 0
                if (fom+1)>(current_fom+1)*(1+0.1):
                    epoch_list[index_fit_pars.index(i)] = epoch*epoch_step
                    fom_diff_list[index_fit_pars.index(i)] = (fom - current_fom)/current_fom
                    #set the original value back
                    self.model.parameters.set_value(i, 1, current_value)
                    #print(epoch_list)
                    break
                if epoch == max_epoch:
                    fom_diff_list[index_fit_pars.index(i)] = (fom - current_fom)/current_fom
                    epoch_list[index_fit_pars.index(i)] = epoch*epoch_step
                    self.model.parameters.set_value(i, 1, current_value)
        sensitivity = np.array(fom_diff_list)/np.array(epoch_list)
        return sensitivity/max(sensitivity)

    def optimize(self, signal = None, signal_fitended = None):
        '''
        Method implementing the main loop of the differential evolution
        algorithm. Note that this method does not run in a separate thread.
        For threading use start_fit, stop_fit and resume_fit instead.
        '''
        # global model
        # self.model = model
        #accumulated speed: adding up all speed for each gengeration
        #self.accum_speed/total_generation is the average speed
        self.accum_speed = 0
        if not MPI_RUN:
            self.text_output('Calculating start FOM ...')
            print('Calculating start FOM ...')
            logging.root.info('Calculating start FOM ...')
            self.running = True
            self.error = False
            self.n_fom = 0
            #print self.pop_vec
            #eval_fom()
            #self.fom_vec = self.trial_fom[:]
            # Old leftovers before going parallel
            self.fom_vec = [self.calc_fom(vec) for vec in self.pop_vec]
            [self.par_evals.append(vec, axis = 0)\
                        for vec in self.pop_vec]
            [self.fom_evals.append(vec) for vec in self.fom_vec]
            #print self.fom_vec
            best_index = argmin(self.fom_vec)
            #print self.fom_vec
            #print best_index
            self.best_vec = copy(self.pop_vec[best_index])
            #print self.best_vec
            self.best_fom = self.fom_vec[best_index]
            #print self.best_fom
            if len(self.fom_log) == 0:
                self.fom_log = r_[self.fom_log,\
                                    [[len(self.fom_log),self.best_fom]]]
            # Flag to keep track if there has been any improvemnts
            # in the fit - used for updates
            self.new_best = True

            self.text_output('Going into optimization ...')
            print('Going into optimization ...')
            logging.root.info('Going into optimization ...')
            # print('optimize: meta weight_factor = ',self.model.fom_func.__weight__.get_weight_factor())

            # Update the plot data for any gui or other output
            self.plot_output(self)
            self.parameter_output(self)

            # Just making gen live in this scope as well...
            gen = self.fom_log[-1,0]
            for gen in range(int(self.fom_log[-1,0]) + 1, self.max_gen\
                                    + int(self.fom_log[-1,0]) + 1):
                if self.stop:
                    #print('stop here!')
                    if signal_fitended != None:
                        signal_fitended.emit('The model run is forced to stop by the user!')
                        break

                t_start = time.time()
                self.init_new_generation(gen)

                # Create the vectors who will be compared to the
                # population vectors
                [self.create_trial(index) for index in range(self.n_pop)]
                # global model
                # self.model = model
                #print(self.trial_vec[0])
                self.eval_fom()
                # print('inside optimize: meta weight_factor = ',self.model.fom_func.__weight__.get_weight_factor())
                # Calculate the fom of the trial vectors and update the population
                [self.update_pop(index) for index in range(self.n_pop)]

                # Add the evaluation to the logging
                #self.par_evals = append(self.par_evals, self.trial_vec, axis = 0)
                [self.par_evals.append(vec, axis = 0)\
                        for vec in self.trial_vec]
                #self.fom_evals = append(self.fom_evals, self.trial_fom)
                [self.fom_evals.append(vec) for vec in self.trial_fom]
                # Add the best value to the fom log
                self.fom_log = r_[self.fom_log,\
                                    [[len(self.fom_log),self.best_fom]]]

                # print(self.fom_log)
                # Let the model calculate the simulation of the best.
                sim_fom = self.calc_sim(self.best_vec)
                # print(sim_fom)
                # Sanity of the model does the simualtions fom agree with
                # the best fom
                if gen == int(self.fom_log[-1,0]) + 1:
                    if abs(sim_fom - self.best_fom) > self.fom_allowed_dis:
                        self.text_output('Disagrement between two different fom'
                                        ' evaluations')
                        self.error = ('The disagreement between two subsequent '
                                    'evaluations is larger than %s. Check the '
                                    'model for circular assignments.'
                                    %self.fom_allowed_dis)
                        if signal_fitended!=None:
                            signal_fitended.emit(self.error)
                            break

                # Update the plot data for any gui or other output
                self.plot_output(self)
                self.parameter_output(self)
                # Time measurent to track the speed
                t = time.time() - t_start
                if t > 0:
                    speed = self.n_pop/t
                else:
                    speed = 999999
                self.accum_speed = self.accum_speed + speed
                outputtext = 'FOM: %.3f Generation: %d Speed: %.1f, Avg. Speed: %.1f'%\
                                            (self.best_fom, gen, speed, self.accum_speed/gen)

                self.new_best = False
                save_tag = False
                # Do an autosave if activated and the interval is coorect
                if gen%self.autosave_interval == 0 and self.use_autosave:
                    save_tag = True
                signal.emit(outputtext, self.model, save_tag)
            signal_fitended.emit('The model run is finished!')
            signal.emit(outputtext, self.model, True)
            if not self.error:
                logging.root.info('Stopped at Generation: %d after %d fom evaluations...'%(gen, self.n_fom))

            # Lets clean up and delete our pool of workers
            if self.use_parallel_processing:
                self.dismount_parallel()
            self.eval_fom = None

            # Now the optimization has stopped
            self.running = False

            # Run application specific clean-up actions
            self.fitting_ended(self)
        else:
            self.text_output('Calculating start FOM ...')
            self.running = True
            self.error = False
            self.n_fom = 0
            #print self.pop_vec
            #eval_fom()
            #self.fom_vec = self.trial_fom[:]
            # Old leftovers before going parallel
            self.fom_vec = [self.calc_fom(vec) for vec in self.pop_vec]
            [self.par_evals.append(vec, axis = 0)\
                        for vec in self.pop_vec]
            [self.fom_evals.append(vec) for vec in self.fom_vec]
            #print self.fom_vec
            best_index = argmin(self.fom_vec)
            #print self.fom_vec
            #print best_index
            self.best_vec = copy(self.pop_vec[best_index])
            #print self.best_vec
            self.best_fom = self.fom_vec[best_index]
            #print self.best_fom
            if len(self.fom_log) == 0:
                self.fom_log = r_[self.fom_log,\
                                    [[len(self.fom_log),self.best_fom]]]
            # Flag to keep track if there has been any improvemnts
            # in the fit - used for updates
            self.new_best = True

            self.text_output('Going into optimization ...')

            # Update the plot data for any gui or other output
            self.plot_output(self)
            self.parameter_output(self)

            # Just making gen live in this scope as well...
            gen = self.fom_log[-1,0]
            for gen in range(int(self.fom_log[-1,0]) + 1, self.max_gen\
                                    + int(self.fom_log[-1,0]) + 1):
                if self.stop:
                    break
                if rank==0:
                    t_start = time.time()

                self.init_new_generation(gen)

                # Create the vectors who will be compared to the
                # population vectors
                if rank==0:
                    [self.create_trial(index) for index in range(self.n_pop)]
                    tmp_trial_vec=self.trial_vec
                else:
                    tmp_trial_vec=0
                tmp_trial_vec=comm.bcast(tmp_trial_vec,root=0)
                self.trial_vec=tmp_trial_vec
                self.eval_fom()
                # Calculate the fom of the trial vectors and update the population
                if rank==0:
                    [self.update_pop(index) for index in range(self.n_pop)]

                    # Add the evaluation to the logging
                    #self.par_evals = append(self.par_evals, self.trial_vec, axis = 0)
                    [self.par_evals.append(vec, axis = 0)\
                            for vec in self.trial_vec]
                    #self.fom_evals = append(self.fom_evals, self.trial_fom)
                    [self.fom_evals.append(vec) for vec in self.trial_fom]

                    # Add the best value to the fom log
                    self.fom_log = r_[self.fom_log,\
                                        [[len(self.fom_log),self.best_fom]]]

                    # Let the model calculate the simulation of the best.
                    sim_fom = self.calc_sim(self.best_vec)

                    # Sanity of the model does the simualtions fom agree with
                    # the best fom
                    if abs(sim_fom - self.best_fom) > self.fom_allowed_dis:
                        self.text_output('Disagrement between two different fom'
                                        ' evaluations')
                        self.error = ('The disagreement between two subsequent '
                                    'evaluations is larger than %s. Check the '
                                    'model for circular assignments.'
                                    %self.fom_allowed_dis)
                        break

                    # Update the plot data for any gui or other output
                    self.plot_output(self)
                    self.parameter_output(self)

                    # Let the optimization sleep for a while
                    time.sleep(self.sleep_time)

                    # Time measurent to track the speed
                    t = time.time() - t_start
                    if t > 0:
                        speed = self.n_pop/t
                    else:
                        speed = 999999
                    self.text_output('FOM: %.3f Generation: %d Speed: %.1f'%\
                                        (self.best_fom, gen, speed))

                    self.new_best = False
                    # Do an autosave if activated and the interval is coorect
                    if gen%self.autosave_interval == 0 and self.use_autosave:
                        self.autosave()

                if rank==0:
                    if not self.error:
                        self.text_output('Stopped at Generation: %d after %d fom evaluations...'%(gen, self.n_fom))

            # Lets clean up and delete our pool of workers

            self.eval_fom = None

            # Now the optimization has stopped
            self.running = False

            # Run application specific clean-up actions
            self.fitting_ended(self)

    def calc_fom(self, vec):
        '''
        Function to calcuate the figure of merit for parameter vector
        vec.
        '''

        # Set the parameter values
        #map(lambda func, value:func(value), self.par_funcs, vec)
        for fun,fun_link,each_vec in zip(self.par_funcs,self.par_funcs_link,vec):
            fun(each_vec)
            if fun_link !=None:
                fun_link(each_vec)
        fom = self.model.evaluate_fit_func()
        # print('diffev: meta weight_factor in calc_fom= ',self.model.fom_func.__weight__.get_weight_factor())
        self.n_fom += 1
        return fom

    def calc_trial_fom(self):
        '''
        Function to calculate the fom values for the trial vectors
        '''
        self.trial_fom = [self.calc_fom(vec) for vec in self.trial_vec]

    def calc_sim(self, vec):
        ''' calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        '''
        # Set the paraemter values
        #for each in self.par_funcs:
        #    print(each.__name__)
        #map(lambda func, value:func(value), self.par_funcs, vec)
        for fun, fun_link,each_vec in zip(self.par_funcs,self.par_funcs_link,vec):
            fun(each_vec)
            if fun_link!=None:
                fun_link(each_vec)

        self.model.evaluate_sim_func()
        return self.model.fom

    def setup_parallel(self):
        '''setup_parallel(self) --> None

        setup for parallel proccesing. Creates a pool of workers with
        as many cpus there is available
        '''
        '''
        self.pool = processing.Pool(processes = self.processes,\
                       initializer = parallel_init,\
                       initargs = (self.model.pickable_copy(), ))
        '''
        print(self.processes,"Processors!")
        self.text_output("Starting a pool with %i workers ..."%\
                            (self.processes, ))
        self._make_task_map()
        self.streaming_actors = [mpi_engine_ray.remote(self.model.pickable_copy()) for _ in range(self.processes)]
        
        #time.sleep()

    def setup_parallel_mpi(self):
        '''setup_parallel(self) --> None

        setup for parallel proccesing. Creates a pool of workers with
        as many cpus there is available
        '''
        if rank==0:
            self.text_output("Starting a pool with %i workers ..."%\
                            (size_group, ))
        #comm=MPI.COMM_WORLD
        #size=comm.Get_size()
        #rank=comm.Get_rank()
        #comm_group= comm.Split(color=rank/(size/self.split_jobs),key= rank)
        #size_group= comm_group.Get_size()
        #rank_group= comm_group.Get_rank()

        parallel_init(self.model.pickable_copy())
        time.sleep(0.1)
        #print "Starting a pool with ", self.processes, " workers ..."

    def dismount_parallel(self):
        ''' dismount_parallel(self) --> None
        Used to close the pool and all its processes
        '''
        ray.shutdown()
        try:
            self.pool.close()
            self.pool.join()
        except:
            pass

        #del self.pool

    def calc_trial_fom_parallel_mpi(self):
        '''calc_trial_fom_parallel(self) --> None

        Function to calculate the fom in parallel using the pool
        '''
        step_len=int(len(self.trial_vec)/size_group)
        remainder=int(len(self.trial_vec)%size_group)
        left,right=0,0
        if rank_group<=remainder-1:
            left=rank_group*(step_len+1)
            right=(rank_group+1)*(step_len+1)-1
        elif rank_group>remainder-1:
            left=remainder*(step_len+1)+(rank_group-remainder)*step_len
            right=remainder*(step_len+1)+(rank_group-remainder+1)*step_len-1
        fom_temp=[]
        for i in range(left,right+1):
            fom_temp.append(parallel_calc_fom(self.trial_vec[i]))
        self.trial_fom=fom_temp

    def _make_task_map(self):
        extra_tasks = len(self.trial_vec)%self.processes
        tasks_even_portion = int(len(self.trial_vec)/self.processes)
        tasks_list = [tasks_even_portion for i in range(self.processes)]
        for i in range(extra_tasks):
            tasks_list[i] += 1
        tasks_list = np.cumsum([0]+tasks_list)
        task_map = {}
        for i in range(self.processes):
            task_map[i] = [tasks_list[i],tasks_list[i+1]]
        self.task_map = task_map

    def _assign_task(self, which):
        for key, value in self.task_map.items():
            if value[1]>which>=value[0]:
                return key

    def calc_trial_fom_parallel(self):
        '''calc_trial_fom_parallel(self) --> None

        Function to calculate the fom in parallel using the pool
        '''
        #self.trial_fom = self.pool.map(parallel_calc_fom, self.trial_vec)
        #self.n_fom += len(self.trial_vec) 
        # [actor.reset_fom.remote() for actor in self.streaming_actors]
        for i in range(self.processes):
            self.streaming_actors[i].calc_fom.remote(self.trial_vec[self.task_map[i][0]:self.task_map[i][1]])
        '''
        for i in range(len(self.trial_vec)):
            self.streaming_actors[self._assign_task(i)].calc_fom.remote(self.trial_vec[i])
        '''
        #results = ray.get([actor.get_fom.remote() for actor in self.streaming_actors])
        foms = []
        for actor in self.streaming_actors:
            foms = foms + ray.get(actor.get_fom.remote())
        '''
        foms = []
        for each in results:
            foms = foms + each
        self.trial_fom = foms
        '''
        #self.trial_fom = np.array(results).flatten()
        self.trial_fom = foms
        # print(len(self.trial_fom))
        self.n_fom += len(self.trial_vec)

    def calc_error_bar(self, index, fom_level):
        '''calc_error_bar(self, parameter) --> (error_bar_low, error_bar_high)

        Calculates the errorbar for one parameter number index.
        returns a float tuple with the error bars. fom_level is the
        level which is the upperboundary of the fom is allowed for the
        calculated error.
        '''
        #print self.par_evals.shape, self.par_evals
        #print self.fom_evals.shape, self.fom_evals
        if self.setup_ok: #and len(self.par_evals) != 0:
            par_values = self.par_evals[:,index]
            #print (self.fom_evals < fom_level).sum()
            #print len(self.fom_evals[:])
            values_under_level = compress(self.fom_evals[:] <\
                                    fom_level*self.best_fom, par_values)
            #print values_under_level
            error_bar_low = values_under_level.min() - self.best_vec[index]
            error_bar_high = values_under_level.max() - self.best_vec[index]
            return (error_bar_low, error_bar_high)
        else:
            raise ErrorBarsError()

    def init_new_generation(self, gen):
        ''' Function that is called every time a new generation starts'''
        pass

    def standard_init_new_generation(self, gen):
        ''' Function that is called every time a new generation starts'''
        pass


    def standard_update_pop(self, index):
        '''
        Function to update population vector index. calcs the figure of merit
        and compares it to the current population vector and also checks
        if it is better than the current best.
        '''
        #fom = self.calc_fom(self.trial_vec[index])
        fom = self.trial_fom[index]
        if fom < self.fom_vec[index]:
            self.pop_vec[index] = self.trial_vec[index].copy()
            self.fom_vec[index] = fom
            if fom < self.best_fom:
                self.new_best = True
                self.best_vec = self.trial_vec[index].copy()
                self.best_fom = fom

    def simplex_old_init_new_generation(self, gen):
        '''It will run the simplex method every simplex_interval
             generation with a fracitonal step given by simple_step
             on the best indivual as well a random fraction of simplex_n individuals.
        '''
        print('Inits new generation')
        if gen%self.simplex_interval == 0:
            spread = array(self.trial_vec).max(0) - array(self.trial_vec).min(0)
            simp = Simplex(self.calc_fom, self.best_vec, spread*self.simplex_step)
            print('Starting simplex run for best vec')
            new_vec, err, iter = simp.minimize(epsilon = self.best_fom/self.simplex_rel_epsilon, maxiters = self.simplex_max_iter)
            print('FOM improvement: ', self.best_fom - err)

            if self.use_boundaries:
                # Check so that the parameters lie indside the bounds
                ok = bitwise_and(self.par_max > new_vec, self.par_min < new_vec)
                # If not inside make a random re-initilazation of that parameter
                new_vec = where(ok, new_vec, random.rand(self.n_dim)*\
                              (self.par_max - self.par_min) + self.par_min)

            new_fom = self.calc_fom(new_vec)
            if new_fom < self.best_fom:
                self.best_fom = new_fom
                self.best_vec = new_vec
                self.pop_vec[0] = new_vec
                self.fom_vec[0] = self.best_fom
                self.new_best = True

            # Apply the simplex to a simplex_n memebers (0-1)
            for index1 in random_mod.sample(xrange(len(self.pop_vec)),
                                   int(len(self.pop_vec)*self.simplex_n)):
                print('Starting simplex run for member: ', index1)
                mem = self.pop_vec[index1]
                mem_fom = self.fom_vec[index1]
                simp = Simplex(self.calc_fom, mem, spread*self.simplex_step)
                new_vec, err, iter = simp.minimize(epsilon = self.best_fom/self.simplex_rel_epsilon, maxiters = self.simplex_max_iter)
                if self.use_boundaries:
                    # Check so that the parameters lie indside the bounds
                    ok = bitwise_and(self.par_max > new_vec, self.par_min < new_vec)
                    # If not inside make a random re-initilazation of that parameter
                    new_vec = where(ok, new_vec, random.rand(self.n_dim)*\
                                    (self.par_max - self.par_min) + self.par_min)

                new_fom = self.calc_fom(new_vec)
                if new_fom < mem_fom:
                    self.pop_vec[index1] = new_vec
                    self.fom_vec[index1] = new_fom
                    if new_fom < self.best_fom:
                        self.best_fom = new_fom
                        self.best_vec = new_vec
                        self.new_best = True

    def simplex_init_new_generation(self, gen):
        '''It will run the simplex method every simplex_interval
             generation with a fracitonal step given by simple_step
             on the simplex_n*n_pop best individuals.
        '''
        print('Inits new generation')
        if gen%self.simplex_interval == 0:
            spread = array(self.trial_vec).max(0) - array(self.trial_vec).min(0)

            indices = argsort(self.fom_vec)
            n_ind = int(self.n_pop*self.simplex_n)
            if n_ind == 0:
                n_ind = 1
            # Apply the simplex to a simplex_n memebers (0-1)
            for index1 in indices[:n_ind]:
                self.text_output('Starting simplex run for member: %d'%index1)
                mem = self.pop_vec[index1].copy()
                mem_fom = self.fom_vec[index1]
                simp = Simplex(self.calc_fom, mem, spread*self.simplex_step)
                new_vec, err, iter = simp.minimize(epsilon = self.best_fom/self.simplex_rel_epsilon, maxiters = self.simplex_max_iter)
                if self.use_boundaries:
                    # Check so that the parameters lie indside the bounds
                    ok = bitwise_and(self.par_max > new_vec, self.par_min < new_vec)
                    # If not inside make a random re-initilazation of that parameter
                    new_vec = where(ok, new_vec, random.rand(self.n_dim)*\
                                    (self.par_max - self.par_min) + self.par_min)

                new_fom = self.calc_fom(new_vec)
                if new_fom < mem_fom:
                    self.pop_vec[index1] = new_vec.copy()
                    self.fom_vec[index1] = new_fom
                    if new_fom < self.best_fom:
                        self.best_fom = new_fom
                        self.best_vec = new_vec.copy()
                        self.new_best = True

    def simplex_best_1_bin(self, index):
        return self.best_1_bin(index)


    def jade_update_pop(self, index):
        ''' A modified update pop to handle the JADE variation of Differential evoluion'''
        fom = self.trial_fom[index]
        if fom < self.fom_vec[index]:
            self.pop_vec[index] = self.trial_vec[index].copy()
            self.fom_vec[index] = fom
            self.updated_kr.append(self.kr_vec[index])
            self.updated_km.append(self.km_vec[index])
            if fom < self.best_fom:
                self.new_best = True
                self.best_vec = self.trial_vec[index].copy()
                self.best_fom = fom


    def jade_init_new_generation(self, gen):
        ''' A modified generation update for jade'''
        #print 'inits generation: ', gen, self.n_pop
        if gen > 1:
            updated_kms = array(self.updated_km)
            updated_krs = array(self.updated_kr)
            if len(updated_kms) != 0:
                self.km = (1.0 - self.c)*self.km + self.c*sum(updated_kms**2)/sum(updated_kms)
                self.kr = (1.0 - self.c)*self.kr + self.c*mean(updated_krs)
        self.km_vec = abs(self.km + random.standard_cauchy(self.n_pop)*0.1)
        self.kr_vec = self.kr + random.normal(size = self.n_pop)*0.1
        #print self.km_vec, self.kr_vec
        print('km: ', self.km, ', kr: ', self.kr)
        #self.km_vec = (self.km_vec >= 1)*1 + (self.km_vec < 1)*self.km_vec
        self.km_vec = where(self.km_vec > 0, self.km_vec, 0)
        self.km_vec = where(self.km_vec < 1, self.km_vec, 1)
        self.kr_vec = where(self.kr_vec > 0, self.kr_vec, 0)
        self.kr_vec = where(self.kr_vec < 1, self.kr_vec, 1)

        self.updated_kr = []
        self.updated_km = []


    def jade_best(self, index):
        vec = self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*len(self.par_evals))
        # Make sure it is not the same vector
        #while index2 == index1:
        #    index2 = int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the best/1 scheme
        #print len(self.km_vec), index, len(self.par_evals),  index2
        mut_vec = vec + self.km_vec[index]*(self.best_vec - vec) + self.km_vec[index]*(self.pop_vec[index1]\
         - self.par_evals[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine = random.rand(self.n_dim) < self.kr_vec[index]
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial = where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        self.trial_vec[index] = trial
        #return trial

    def best_1_bin(self, index):
        '''best_1_bin(self, vec) --> trial [1D array]

        The default create_trial function for this class.
        uses the best1bin method to create a new vector from the population.
        '''
        vec = self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the best/1 scheme
        mut_vec = self.best_vec + self.km*(self.pop_vec[index1]\
         - self.pop_vec[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine = random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial = where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        self.trial_vec[index] = trial
        #return trial


    def best_either_or(self, index):
        '''best_either_or(self, vec) --> trial [1D array]

        The either/or scheme for creating a trial. Using the best vector
        as base vector.
        '''
        vec = self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)

        if random.rand(1) < self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial = self.best_vec + self.km*(self.pop_vec[index1]\
            - self.pop_vec[index2])
        else:
            # Trying something else out more like normal recombination
            trial = vec + self.kr*(self.pop_vec[index1]\
            + self.pop_vec[index2] - 2*vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        self.trial_vec[index] = trial
        #return trial

    def rand_1_bin(self, index):
        '''best_1_bin(self, vec) --> trial [1D array]

        The default create_trial function for this class.
        uses the best1bin method to create a new vector from the population.
        '''
        vec = self.pop_vec[index]
        # Create mutation vector
        # Select three random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
        index3 = int(random.rand(1)*self.n_pop)
        while index3 == index1 or index3 == index2:
            index3 = int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the rand/1 scheme
        mut_vec = self.pop_vec[index3] + self.km*(self.pop_vec[index1]\
         - self.pop_vec[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine = random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial = where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        self.trial_vec[index] = trial
        #return trial

    def rand_either_or(self, index):
        '''rand_either_or(self, vec) --> trial [1D array]

        random base vector either/or trial scheme
        '''
        vec = self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
        index0 = int(random.rand(1)*self.n_pop)
        while index0 == index1 or index0 == index2:
            index0 = int(random.rand(1)*self.n_pop)

        if random.rand(1) < self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial = self.pop_vec[index0] + self.km*(self.pop_vec[index1]\
            - self.pop_vec[index2])
        else:
            # Calculate a continous recomibination
            # Trying something else out more like normal recombination
            trial = self.pop_vec[index0] + self.kr*(self.pop_vec[index1]\
            + self.pop_vec[index2] - 2*self.pop_vec[index0])
            #trial = vec + self.kr*(self.pop_vec[index1]\
            #        + self.pop_vec[index2] - 2*vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        self.trial_vec[index] = trial
        #return trial


    # Different function for acessing and setting parameters that
    # the user should have control over.

    def set_text_output_func(self, func):
        '''set_text_output_func(self, func) --> None

        Set the output function for the text output from the optimizer.
        Should be a function that takes a string as input argument.
        The default function is a simple print statement.
        '''
        self.text_output = func

    def set_plot_output_func(self, func):
       '''set_plot_output_func(self, func) --> None

       Set the output function for the plot output from the optimizer.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.plot_output = func


    def set_parameter_output_func(self, func):
       '''set_parameters_output_func(self, func) --> None

       Set the output function for the parameters output from the optimizer.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.parameter_output = func


    def set_fitting_ended_func(self, func):
       '''set_fitting_ended_func(self, func) --> None

       Set the function when the optimizer has finsihed the fitting.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.fitting_ended = func

    def set_autosave_func(self, func):
        '''set_autosave_func(self, func) --> None

        Set the function that the optimizer uses to do an autosave
        of the current fit. Function func should not take any arguments.
        '''
        self.autosave = func

    # Some get functions

    def get_model(self):
        '''get_model(self) --> model
        Getter that returns the model in use in solver.
        '''
        return self.model

    def get_fom_log(self):
        '''get_fom_log(self) -->  fom [array]
        Returns the fom as a fcn of iteration in an array.
        Last element last fom value
        '''
        return array(self.fom_log)

    def get_create_trial(self, index = False):
        '''get_create_trial(self, index = False) --> string or int

        returns the current create trial function name if index is False as
        a string or as index in the mutation_schemes list.
        '''
        pos = self.mutation_schemes.index(self.create_trial)
        if index:
            # return the position
            return pos
        else:
            # return the name
            return self.mutation_schemes[pos].__name__

    def set_km(self, val):
        '''set_km(self, val) --> None
        '''
        self.km = val

    def set_kr(self, val):
        '''set_kr(self, val) --> None
        '''
        self.kr = val

    def set_create_trial(self, val):
        '''set_create_trial(self, val) --> None

        Raises LookupError if the value val [string] does not correspond
        to a mutation scheme/trial function
        '''
        # Get the names of the available functions
        names = [f.__name__ for f in self.mutation_schemes]
        # Find the postion of val

        pos = names.index(val)
        self.create_trial = self.mutation_schemes[pos]
        if val == 'jade_best':
            self.update_pop = self.jade_update_pop
            self.init_new_generation = self.jade_init_new_generation
        elif val == 'simplex_best_1_bin':
            self.init_new_generation = self.simplex_init_new_generation
            self.update_pop = self.standard_update_pop
        else:
            self.init_new_generation = self.standard_init_new_generation
            self.update_pop = self.standard_update_pop

    def set_pop_mult(self, val):
        '''set_pop_mult(self, val) --> None
        '''
        self.pop_mult = val

    def set_pop_size(self, val):
        '''set_pop_size(self, val) --> None
        '''
        self.pop_size = int(val)

    def set_max_generations(self, val):
        '''set_max_generations(self, val) --> None
        '''
        self.max_generations = int(val)

    def set_max_generation_mult(self, val):
        '''set_max_generation_mult(self, val) --> None
        '''
        self.max_generation_mult = val

    def set_sleep_time(self, val):
        '''set_sleep_time(self, val) --> None
        '''
        self.sleep_time = val

    def set_max_log(self, val):
        '''Sets the maximum number of logged elements
        '''
        self.max_log = val

    def set_use_pop_mult(self, val):
        '''set_use_pop_mult(self, val) --> None
        '''
        self.use_pop_mult = val

    def set_use_max_generations(self, val):
        '''set_use_max_generations(self, val) --> None
        '''
        self.use_max_generations = val

    def set_use_start_guess(self, val):
        '''set_use_start_guess(self, val) --> None
        '''
        self.use_start_guess = val

    def set_use_boundaries(self, val):
        '''set_use_boundaries(self, val) --> None
        '''
        self.use_boundaries = val

    def set_use_autosave(self, val):
        '''set_use_autosave(self, val) --> None
        '''
        self.use_autosave = val

    def set_autosave_interval(self, val):
        '''set_autosave_interval(self, val) --> None
        '''
        self.autosave_interval = int(val)

    def set_use_parallel_processing(self, val):
        '''set_use_parallel_processing(self, val) --> None
        '''
        if __parallel_loaded__:
            self.use_parallel_processing = val
        else:
            self.use_parallel_processing = False

    def set_processes(self, val):
        '''set_processes(self, val) --> None
        '''
        self.processes = int(val)

    def set_chunksize(self, val):
        '''set_chunksize(self, val) --> None
        '''
        self.chunksize = int(val)

    def set_fom_allowed_dis(self, val):
        '''set_chunksize(self, val) --> None
        '''
        self.fom_allowed_dis = float(val)

@ray.remote
class mpi_engine_ray(object):
    def __init__(self, model):
        self.model = model
        self.model._reset_module()
        self.model.simulate()
        self.fom = []
        self.par_funcs, self.start_guess, self.par_min, self.par_max, self.par_funcs_link = self.model.get_fit_pars()

    def calc_fom(self, vecs):
        self.reset_fom()
        for vec in vecs:
            for i in range(len(self.par_funcs)):
                self.par_funcs[i](vec[i])
                if self.par_funcs_link[i]!=None:
                    self.par_funcs_link[i](vec[i])
            # evaluate the model and calculate the fom
            self.fom.append(self.model.evaluate_fit_func())
            #self.fom.append(fom)

    def get_fom(self):
        return self.fom

    def reset_fom(self):
        self.fom = []

class fit_model_NLLS(object):
    def __init__(self, model):
        self.model = model
        self.kwargs = {}
        self.fom = None
        self.running = False

    def retrieve_fit_pars(self):
        self.model._reset_module()
        self.model.simulate()
        self.par_funcs, self.start_guess, self.par_min, self.par_max, self.par_funcs_link = self.model.get_fit_pars()
        _,_,self.pars_init,left, right, _ = self.model.parameters.get_fit_pars() 
        #intentionally move the best fit par values away by 2% to a better statistic error estimation from NLLS fit
        self.pars_init = np.array(self.pars_init)*1.02
        self.bounds = (left, right)
        self.x = np.zeros(sum([len(each.x) for each in self.model.data]))

    def fit_model(self, cb = None):
        self.running = True
        self.run_num = 0
        self.retrieve_fit_pars()
        self.fom = None
        def fit_func(x, *vec):
            for i in range(len(self.par_funcs)):
                self.par_funcs[i](vec[i])
                if self.par_funcs_link[i]!=None:
                    self.par_funcs_link[i](vec[i])
            fom = self.model.evaluate_fit_func()
            self.run_num = self.run_num + 1
            if self.run_num > 2000:
                self.running = False
                if cb:
                    cb.emit()
            self.fom = fom
            return fom
        self.popt, self.pcov = curve_fit(fit_func, self.x, self.x*0,p0=self.pars_init,method="trf",loss='cauchy',maxfev=2000,ftol=1e-8,**self.kwargs)
        # self.popt, self.pcov = curve_fit(fit_func, self.x, self.x*0,p0=self.pars_init,maxfev=10000,**self.kwargs)
        self.running = False
        # popt, pcov = curve_fit(fit_func, 0, 0.,p0=self.pars_init,method="trf",loss="linear",bounds=self.bounds,max_nfev=1000,verbose=2,ftol=1e-8)
        self.perr = np.sqrt(np.diag(self.pcov))

#==============================================================================
# Functions that is needed for parallel processing!

def parallel_init(model_copy):
    '''parallel_init(model_copy) --> None

    parallel initilization of a pool of processes. The function takes a
    pickle safe copy of the model and resets the script module and the compiles
    the script and creates function to set the variables.
    '''
    global model, par_funcs, par_funcs_link
    model = model_copy
    model._reset_module()
    model.simulate()
    (par_funcs, start_guess, par_min, par_max, par_funcs_link) = model.get_fit_pars()

def parallel_calc_fom(vec):
    '''parallel_calc_fom(vec) --> fom (float)

    function that is used to calculate the fom in a parallel process.
    It is a copy of calc_fom in the DiffEv class
    '''
    global model, par_funcs, par_funcs_link
    for i in range(len(par_funcs)):
        par_funcs[i](vec[i])
        if par_funcs_link[i]!=None:
            par_funcs_link[i](vec[i])
    # evaluate the model and calculate the fom
    fom = model.evaluate_fit_func()

    return fom


#==============================================================================
def default_text_output(text):
    sys.stdout.flush()

def default_plot_output(solver):
    pass

def default_parameter_output(solver):
    pass

def default_fitting_ended(solver):
    pass

def defualt_autosave():
    pass

def _calc_fom(model, vec, par_funcs):
        '''
        Function to calcuate the figure of merit for parameter vector
        vec.
        '''
        # Set the paraemter values
        map(lambda func, value:func(value), par_funcs, vec)

        return model.evaluate_fit_func()

#==============================================================================
# BEGIN: class CircBuffer
class CircBuffer:
    '''A buffer with a fixed length to store the logging data from the diffev
    class. Initilized to a maximumlength after which it starts to overwrite
    the data again.
    '''
    def __init__(self, maxlen, buffer = None):
        '''Inits the class with a certain maximum length maxlen.
        '''
        self.maxlen = int(maxlen)
        self.pos = -1
        self.filled = False
        if buffer == None:
            self.buffer = zeros((self.maxlen,))
        else:
            if len(buffer) != 0:
                self.buffer = array(buffer).repeat(
                    ceil(self.maxlen/(len(buffer)*1.0)), 0)[:self.maxlen]
                self.pos = len(buffer) - 1
            else:
               self.buffer = zeros((self.maxlen,) + buffer.shape[1:])


    def reset(self, buffer = None):
        '''Resets the buffer to the initial state
        '''
        self.pos = -1
        self.filled = False
        #self.buffer = buffer
        if buffer == None:
             self.buffer = zeros((self.maxlen,))
        else:
            if len(buffer) != 0:
                self.buffer = array(buffer).repeat(
                    ceil(self.maxlen/(len(buffer)*1.0)), 0)[:self.maxlen]
                self.pos = len(buffer) - 1
            else:
                self.buffer = zeros((self.maxlen,) + buffer.shape[1:])



    def append(self, item, axis = None):
        '''Appends an element to the last position of the buffer
        '''
        new_pos = (self.pos + 1)%self.maxlen
        if len(self.buffer) >= self.maxlen:
            if self.pos >= (self.maxlen - 1):
                self.filled = True
            self.buffer[new_pos] = array(item).real
        else:
            self.buffer = append(self.buffer, item, axis = axis)
        self.pos = new_pos

    def array(self):
        '''returns an ordered array instead of the circular
        working version
        '''
        if self.filled:
            return r_[self.buffer[self.pos+1:], self.buffer[:self.pos+1]]
        else:
            return r_[self.buffer[:self.pos+1]]

    def copy_from(self, object):
        '''Add copy support
        '''
        if type(object) == type(array([])):
            self.buffer = object[-self.maxlen:]
        elif object.__class__ == self.__class__:
            # Check if the buffer has been removed.
            if len(object.buffer) == 0:
                self.__init__(object.maxlen, object.buffer)
            else:
                self.buffer = object.buffer.copy()
                self.maxlen = object.maxlen
                self.pos = object.pos
                try:
                    self.filled = object.filled
                except:
                    self.filled = False
        else:
            raise TypeError('CircBuffer support only copying from CircBuffer'\
                        ' and arrays.')


    def __len__(self):
        if self.filled:
            return len(self.buffer)
        else:
            return (self.pos > 0)*self.pos

    def __getitem__(self, key):
        return self.array().__getitem__(key)



# END: class CircBuffer
#==============================================================================
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ErrorBarError(GenericError):
    '''Error class for the fom evaluation'''
    def __init__(self):
        ''' __init__(self) --> None'''
        #self.error_message = error_message

    def __str__(self):
        text = 'Could not evaluate the error bars. A fit has to be made' +\
                'before they can be calculated'
        return text
