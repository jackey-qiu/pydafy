'''
Library that contains the the class Model. This
is a general class that store and binds togheter all the other
classes that is part of model parameters, such as data and parameters.
Programmer: Matts Bjorck
Last changed: 2008 06 24
'''

# Standard libraries
import shelve, os,zipfile
import types,sys
#import _pickle as pickle
import pickle
from io import StringIO
import pdb, traceback
import numpy as np
# GenX libraries
#import data
import dafy.core.FilterPool.data_superrod as data
from dafy.core.FilterPool import data_superrod
import dafy.core.FilterPool.parameters as parameters
import dafy.core.util.fom_funcs as fom_funcs
from dafy.core.util.fom_funcs import weight_fom_based_on_HKL
# from dafy.core.util.path import superrod_path_list
# for each in superrod_path_list:
    # sys.path.append(each)
#import dill

#==============================================================================
#BEGIN: Class Model

class Model:
    ''' A class that holds the model i.e. the script that defines
        the model and the data + various other attributes.
    '''

    def __init__(self, config = None):
        '''
        Create a instance and init all the varaibles.
        '''
        self.config = config
        #self.data
        self.data = data.DataList()
        self.data_original = data.DataList()
        self.script = ''
        self.parameters = parameters.Parameters()

        #self.fom_func = default_fom_func
        # self.fom_func = fom_funcs.log # The function that evaluates the fom
        self.fom_func = fom_funcs.chi2bars # The function that evaluates the fom
        self.fom = None # The value of the fom function
        self.weight_factor = 1 #fom weighting factor
        self.weight_map = {}#fom weighting factor map
        self.weight_decorator = weight_fom_based_on_HKL

        # Registred classes that is looked for in the model
        self.registred_classes = []
        #self.registred_classes = ['Layer','Stack','Sample','Instrument',\
        #                            'model.Layer', 'model.Stack',\
        #                             'model.Sample','model.Instrument',\
        #                             'UserVars','Surface','Bulk']
        self.set_func = 'set' #'set'
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ''
        self.saved = True
        self.compiled = False

    def read_config(self):
        '''Read in the config file
        '''
        # Ceck so that config is loaded
        if not self.config:
            return
        try:
            val = self.config.get('parameters', 'registred classes')
        except:
            print('Could not find config for parameters, registered classes')
        else:
            self.registred_classes = [s.strip() for s in val.split(';')]
        try:
            val = self.config.get('parameters', 'set func')
        except:
            print('Could not find config for parameters, set func')
        else:
            self.set_func = val


    def load(self,filename):
        '''
        Function to load the necessary parameters from a model file.
        '''
        try:
            loadfile = zipfile.ZipFile(filename, 'r')
        except Exception as e:
            raise IOError('Could not open file.', filename)
        #new_data = pickle.loads(open(loadfile.read('data')))
        #new_data = pickle.loads(loadfile.read('data'),fix_imports=True,encoding = 'latin1')
        # dill._dill._reverse_typemap["ObjectType"] = object
        try:
            new_data = pickle.loads(loadfile.read('data'),fix_imports=True, encoding = 'latin1')
            # new_data = pickle.loads(open(loadfile.read('data'),'rb'),fix_imports=True, encoding = 'latin1')
            #print 'data_type',type(new_data)
            # print('new_data',len(new_data))
            self.data.safe_copy(new_data)
            self.data_original.safe_copy(new_data)
            self.data.concatenate_all_ctr_datasets()
            self.data_original.concatenate_all_ctr_datasets()
        except Exception as e:
            raise IOError('Could not locate the data section.', filename)
        try:
            self.script = pickle.loads(loadfile.read('script'))
            #print 'script_type',type(self.script)
        except Exception as e:
            raise IOError('Could not locate the script.', filename)

        try:
            new_parameters = pickle.loads(loadfile.read('parameters'),encoding='latin1')
            #print dir(new_parameters)
            self.parameters.safe_copy(new_parameters)
        except Exception:
            raise IOError('Could not locate the parameters section.', filename)
        try:
            self.fom_func = pickle.loads(loadfile.read('fomfunction'))
        except Exception:
           raise IOError('Could not locate the fomfunction section.', filename)

        loadfile.close()

        self.filename = os.path.abspath(filename)
        self.saved = True
        self.script_module = types.ModuleType('genx_script_module')
        self.script_module.__dict__['data'] = self.data
        self.compiled = False

    def save_all(self,filename, optimizer):
        '''
        Function to save the model to file filename
        '''
        try:
            savefile = zipfile.ZipFile(filename, 'w')
        except Exception as e:
            raise IOError(str(e), filename)

        # Save the data structures to file
        try:
            #print self.data
            savefile.writestr('data', pickle.dumps(self.data_original))

        except Exception as e:
            raise IOError(str(e), filename)
        try:
            #print self.script
            savefile.writestr('script', pickle.dumps(self.script))
        except Exception as e:
            raise IOError(str(e), filename)
        try:
            #print self.parameters
            savefile.writestr('parameters', pickle.dumps(self.parameters))
        except Exception:
           raise IOError(str(e), filename)
        try:
            #print self.fom_func
            savefile.writestr('fomfunction', pickle.dumps(self.fom_func))
        except Exception:
           raise IOError(str(e), filename)

        #now save optimizer configs
        names = ['k_m','k_r','Figure of merit','Method','Auto save, interval','weighting factor','weighting region','start guess','Generation size','Population size']
        funcs = ['km','kr',"fom_func",'create_trial','autosave_interval','weight_factor','weight_map','use_start_guess','max_generations','pop_size']
        # Check so the filename is ok i.e. has been saved
        if filename == '':
            raise IOError('File must be saved before new information is added'\
                            ,'')

        for name, attr in zip(names, funcs):
            if attr in ['weight_factor','weight_map']:
                text = str(getattr(self,attr))
            elif attr=='create_trial':
                text = getattr(optimizer,attr).__name__
            elif attr == 'fom_func':
                text = getattr(self,attr).__name__
            else:
                text = str(getattr(optimizer,attr))
            try:
                savefile.writestr(name, text)
            except Exception as e:
                raise IOError(str(e), filename)

        savefile.close()

        # self.filename = os.path.abspath(filename)
        self.saved = True

    def save(self,filename):
        '''
        Function to save the model to file filename
        '''
        try:
            savefile = zipfile.ZipFile(filename, 'w')
        except Exception as e:
            raise IOError(str(e), filename)

        # Save the data structures to file
        try:
            #print self.data
            savefile.writestr('data', pickle.dumps(self.data_original))

        except Exception as e:
            raise IOError(str(e), filename)
        try:
            #print self.script
            savefile.writestr('script', pickle.dumps(self.script))
        except Exception as e:
            raise IOError(str(e), filename)
        try:
            #print self.parameters
            savefile.writestr('parameters', pickle.dumps(self.parameters))
        except Exception:
           raise IOError(str(e), filename)
        try:
            #print self.fom_func
            savefile.writestr('fomfunction', pickle.dumps(self.fom_func))
        except Exception:
           raise IOError(str(e), filename)

        savefile.close()

        # self.filename = os.path.abspath(filename)
        self.saved = True

    def save_addition(self, name, text):
        '''save_addition(self, name, text) --> None

        save additional text [string] subfile with name name [string]\
         to the current file.
        '''
        # Check so the filename is ok i.e. has been saved
        if self.filename == '':
            raise IOError('File must be saved before new information is added'\
                            ,'')
        try:
            savefile = zipfile.ZipFile(self.filename, 'a')
        except Exception as e:
            raise IOError(str(e), self.filename)

        # Check so the model data is not overwritten
        if name == 'data' or name == 'script' or name == 'parameters':
            raise IOError('It not alllowed to save a subfile with name: %s'%name)

        try:
            if type(text)==str:
                savefile.writestr(name, text)
            else:
                savefile.writestr(name, pickle.dumps(text))
        except Exception as e:
            raise IOError(str(e), self.filename)
        savefile.close()

    #save additional attributes obtained from optimizer
    def save_addition_from_optimizer(self,filename,optimizer):
        '''save_addition(self, name, text) --> None

        save additional text [string] subfile with name name [string]\
         to the current file.
        '''
        names = ['k_m','k_r','Figure of merit','Method','Auto save, interval','weighting factor','weighting region','start guess','Generation size','Population size']
        funcs = ['km','kr',"fom_func",'create_trial','autosave_interval','weight_factor','weight_map','use_start_guess','max_generations','pop_size']
        # Check so the filename is ok i.e. has been saved
        if filename == '':
            raise IOError('File must be saved before new information is added'\
                            ,'')
        try:
            savefile = zipfile.ZipFile(filename, 'a')
        except Exception as e:
            raise IOError(str(e), filename)

        for name, attr in zip(names, funcs):
            if attr in ['weight_factor','weight_map']:
                text = str(getattr(self,attr))
            elif attr=='create_trial':
                text = getattr(optimizer,attr).__name__
            elif attr == 'fom_func':
                text = getattr(self,attr).__name__
            else:
                text = str(getattr(optimizer,attr))
            try:
                savefile.writestr(name, text)
            except Exception as e:
                raise IOError(str(e), filename)
        savefile.close()

    def load_addition(self, name, load_type = 'string'):
        '''load_addition(self, name) --> text

        load additional text [string] subfile with name name [string]\
         to the current model file.
        '''
        # Check so the filename is ok i.e. has been saved
        if self.filename == '':
            raise IOError('File must be loaded before additional '\
                        + 'information is read'\
                            ,'')
        try:
            loadfile = zipfile.ZipFile(self.filename, 'r')
        except Exception as e:
            raise IOError('Could not open the file', self.filename)

        try:
            text = loadfile.read(name)
            if load_type != 'string':
                text = pickle.loads(text,fix_imports=True, encoding = 'latin1')
        except Exception as e:
            raise IOError('Could not read the section named: %s'%name,\
                            self.filename)
        loadfile.close()
        return text

    #load_addition config attributes and update them to the optimizer
    #used in mpi_script
    def apply_addition_to_optimizer(self,optimizer):
        types= [float,float,str,int,float,str,bool,int,int]
        pars = ['k_m','k_r','Method','Auto save, interval','weighting factor','weighting region','start guess','Generation size','Population size']
        funcs = ['set_km','set_kr','set_create_trial','set_autosave_interval','set_weighting_factor','set_weighting_region','set_use_start_guess','set_max_generations','set_pop_size']
        for i in range(len(pars)):
            type_ = types[i]
            if type_ == float:
                value = np.round(float(self.load_addition(pars[i])),2)
            elif type_==str:
                value = self.load_addition(pars[i]).decode("utf-8")
            elif type_==bool:
                value = (self.load_addition(pars[i]).decode("ASCII")=="True")
            else:
                value = type_(self.load_addition(pars[i]))
            if pars[i] == 'weighting factor':
                getattr(self,funcs[i])(value)
            elif pars[i] == 'weighting region':
                getattr(self,funcs[i])(eval(value))
            else:
                getattr(optimizer,funcs[i])(value)
            print(f"setting {pars[i]} = {value} now!")

    def _reset_module(self):
        '''
        Internal method for resetting the module before compilation
        '''
        self.script_module = types.ModuleType('genx_script_module')
        #self.script_module = Temp()
        #self.script_module.__dict__ = {}
        # Bind data for preprocessing with the script
        self.script_module.__dict__['data'] = self.data
        self.compiled = False

    def compile_script(self):
        '''
        compile the script in a seperate module.
        '''

        self._reset_module()
        # Testing to see if this works under windows
        self.script = '\n'.join(self.script.splitlines())
        
        try:
            exec(self.script, self.script_module.__dict__)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise ModelError(str(val), 0)
        else:
            self.compiled = True

    def eval_in_model(self, codestring):
        '''
        Excecute the code in codestring in the namespace of
        model module
        '''
        #exec codestring in self.script_module.__dict__
        # result = eval(codestring, self.script_module.__dict__)
        result = eval(codestring, self.script_module.__dict__)
        # print('Sucessfully evaluted: ', codestring)
        return result

    def calc_fom(self, simulated_data,wt=1,wt_list=[]):
        '''calc_fom(self, fomlist) -> fom_raw (list of arrays),
                                      fom_indiv(list of floats),
                                      fom(float)

        Sums up the evaluation of the fom values calculated for each
         data point to form the overall fom function for all data sets.
        '''
        #fom_raw = self.fom_func(simulated_data, self.data)
        #weight the fom value
        fom_raw = self.weight_decorator(self.weight_factor,self.weight_map)(self.fom_func)(simulated_data,self.data)
        # Sum up a unique fom for each data set in use
        fom_indiv=[]
        if wt_list==[]:
            fom_indiv = [np.sum(np.abs(fom_set)) for fom_set in fom_raw]
        else:
            fom_indiv = [np.sum(np.abs(fom_set))*scale for (fom_set,scale) in zip(fom_raw,wt_list)]
        fom = np.sum([f for f, d in zip(fom_indiv, self.data) if d.use])
        # Lets extract the number of datapoints as well:
        N = np.sum([len(fom_set) for fom_set, d in zip(fom_raw, self.data) if d.use])
        # And the number of fit parameters
        p = self.parameters.get_len_fit_pars()
        #self.fom_dof = fom/((N-p)*1.0)
        try:
            use_dif = self.fom_func.__div_dof__
        except Exception:
            use_dif = False
        if use_dif:
            if N>p:
                fom = fom/((N-p)*1.0)
            else:
                fom=fom/(N*1.0)

        return fom_raw, fom_indiv, fom*wt
        #return fom_raw,fom_indiv,fom_raw[0][3]

    def evaluate_fit_func(self):
        ''' evaluate_fit_func(self) --> fom (float)

        Evalute the Simulation fucntion and returns the fom. Use this one
        for fitting. Use evaluate_sim_func(self) for updating of plots
        and such.
        '''
        try:
            simulated_data,wt,wt_list = self.script_module.Sim(self.data)
            fom_raw, fom_inidv, fom = self.calc_fom(simulated_data,wt,wt_list)
        except:
            simulated_data,wt = self.script_module.Sim(self.data)
            fom_raw, fom_inidv, fom = self.calc_fom(simulated_data,wt)
        #fom = self.fom_func(simulated_data, self.data)
        #fom_raw, fom_inidv, fom = self.calc_fom(simulated_data,wt,wt_list)
        return fom

    def evaluate_sim_func(self):
        '''evaluate_sim_func(self) --> None

        Evalute the Simulation function and updates the data simulated data
        as well as the fom of the model. Use this one for calculating data to
        update plots, simulations and such.
        '''
        try:
            simulated_data,wt,wt_list = self.script_module.Sim(self.data)
        except:
            try:
                simulated_data,wt = self.script_module.Sim(self.data)
            except Exception as e:
                outp = StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                raise ModelError(str(val), 1)

        # check so that the Sim function returns anything
        if not simulated_data:
            text = 'The Sim function does not return anything, it should' +\
            ' return a list of the same length as the number of data sets.'
            raise ModelError(text, 1)
        # Check so the number of data sets is correct
        if len(simulated_data) != len(self.data):
            text = 'The number of simulated data sets returned by the Sim function'\
             + ' has to be same as the number of loaded data sets.\n' +\
             'Number of loaded data sets: ' + str(len(self.data)) +\
             '\nNumber of simulated data sets: ' + str(len(simulated_data))
            raise ModelError(text, 1)

        self.data.set_simulated_data(simulated_data)

        try:
            #self.fom = self.fom_func(simulated_data, self.data)
            fom_raw, fom_inidv, fom = self.calc_fom(simulated_data,wt,wt_list)
            self.fom = fom
        except:
            try:
                fom_raw, fom_inidv, fom = self.calc_fom(simulated_data,wt)
                self.fom = fom
            except Exception as e:
                outp = StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                raise FomError(str(val))
        #print len(fom_raw)
        self.data.set_fom_data(fom_raw)

    def create_fit_func(self, str):
        '''create_fit_func(self, str) --> function

        Creates a function from the string expression in string.
        If the string is a function in the model this function will be
        returned if string represents anything else a function that sets that
        object will be returned.
        '''
        object = self.eval_in_model(str)
        #print type(object)
        # Is it a function or a method!
        name = type(object).__name__
        if name == 'instancemethod' or name == 'function' or name == 'method':
            return object
        # Nope lets make a function of it
        else:
            print(name)
            #print 'def __tempfunc__(val):\n\t%s = val'%str
            #The function must be created in the module in order to acess
            # the different variables
            exec('def __tempfunc__(val):\n\t%s = val'%str\
                in self.script_module.__dict__)

            #print self.script_module.__tempfunc__
            return self.script_module.__tempfunc__

    def get_fit_pars(self):
        ''' get_fit_pars(self) --> (funcs, values, min_values, max_values)

        Returns the parameters used with fitting. i.e. the function to
        set the paraemters, the guess value (values), minimum allowed values
        and the maximum allowed values
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals, sfuncs_link) =\
            self.parameters.get_fit_pars()
        if len(sfuncs) == 0:
            raise ParameterError(sfuncs, 0, 'None', 4)
        """
        # Check for min and max on all the values
        for i in range(len(vals)):
            # parameter less than min
            if vals[i] < minvals[i]:
                raise ParameterError(sfuncs[i], row_numbers[i], 'None', 3)
            # parameter larger than max
            if vals[i] > maxvals[i]:
                raise ParameterError(sfuncs[i], row_numbers[i], 'None', 2)
        """
        # Compile the strings to create the functions..
        funcs = []
        funcs_link = []
        # print(sfuncs)
        for func in sfuncs:
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, row_numbers[len(funcs)], str(e),0)
        for func in sfuncs_link:
            if func=='':
                funcs_link.append(None)
            else:
                try:
                    funcs_link.append(self.create_fit_func(func))
                except Exception as e:
                    raise ParameterError(func, row_numbers[len(funcs_link)], str(e),0)
        return (funcs, vals, minvals, maxvals,funcs_link)

    def get_fit_values(self):
        '''get_fit_values(self) --> values

        Returns the current parameters values that the user has ticked as
        fittable.
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals) =\
            self.parameters.get_fit_pars()
        return vals

    def get_sim_pars(self):
        ''' get_sim_pars(self) --> (funcs, values)

        Returns the parameters used with simulations. i.e. the function to
        set the parameters, the guess value (values). Used for simulation,
        for fitting see get_fit_pars(self).s
        '''
        (sfuncs, vals, sfuncs_link) = self.parameters.get_sim_pars()
        # Compile the strings to create the functions..
        funcs = []
        funcs_link = []
        for func in sfuncs:
            # funcs.append(self.create_fit_func(func))
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, len(funcs), str(e),0)
        #funcs of linked parameters
        for func_link in sfuncs_link:
            if func_link=='':
                funcs_link.append(None)
            else:
                try:
                    funcs_link.append(self.create_fit_func(func_link))
                    # print('Linking {} successfully!'.format(func_link))
                except:
                    funcs_link.append('None')
        return (funcs, vals, funcs_link)

    def simulate(self, compile = True):
        '''simulate(self, compile = True) --> None

        Simulates the data sets using the values given in parameters...
        also compiles the script if asked for (default)
        '''
        if compile:
            self.compile_script()
        (funcs, vals, funcs_link) = self.get_sim_pars()
        # print 'Functions to evulate: ', funcs
        # Set the parameter values in the model
        #[func(val) for func,val in zip(funcs, vals)]
        i = 0
        for func, val, func_link in zip(funcs,vals,funcs_link):
            try:
                func(val)
            except Exception as e:
                (sfuncs_tmp, vals_tmp, sfuncs_link_temp) = self.parameters.get_sim_pars()
                raise ParameterError(sfuncs_tmp[i], i, str(e), 1)
            try:
                if func_link!=None:
                    func_link(val)
            except Exception as e:
                (sfuncs_tmp, vals_tmp, sfuncs_link_temp) = self.parameters.get_sim_pars()
                raise ParameterError(sfuncs_link_temp[i], i, str(e), 1)
            i += 1
        self.evaluate_sim_func()

    def new_model(self):
        '''
        new_model(self) --> None

        Reinitilizes the model. Thus, removes all the traces of the
        previous model.
        '''
        self.data = data.DataList()
        self.script = ''
        self.parameters = parameters.Parameters()

        #self.fom_func = default_fom_func
        self.fom_func = fom_funcs.log
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ''
        self.saved = False

    def pickable_copy(self):
        '''pickable_copy(self) --> model

        Creates a pickable object of the model. Can be used for saving or
        sending to other processes, i.e., parallel processing.
        '''
        model_copy = Model(self.config)
        model_copy.data = self.data
        model_copy.script = self.script
        model_copy.parameters = self.parameters
        model_copy.fom_func = self.fom_func
        model_copy.weight_factor = self.weight_factor
        model_copy.weight_map = self.weight_map
        # The most important stuff - a module is not pickable
        model_copy.script_module = None
        model_copy.filename = self.filename
        model_copy.compiled = self.compiled
        model_copy.fom = self.fom
        model_copy.saved = self.saved
        # print('original',self.fom_func.__weight__.get_weight_factor())
        # print('pickle copy',model_copy.fom_func.__weight__.get_weight_factor())
        return model_copy

    def get_table_as_ascii(self):
        '''get_table_as_ascii(self) --> None

        Just a copy of the parameters class method get_ascii_output()
        '''
        return self.parameters.get_ascii_output()

    def get_data_as_asciitable(self, indices = None):
        '''get_data_as_asciitable(self, indices None) --> string

        Just a copy of the method defined in data with the same name.
        '''
        return self.data.get_data_as_asciitable(indices)

    def export_table(self, filename):
        '''
        Export the table to filename. ASCII output.
        '''
        self._save_to_file(filename, self.parameters.get_ascii_output())

    def export_data(self, basename):
        '''
        Export the data to files with basename filename. ASCII output.
        The fileending will be .dat
        First column is the x-values.
        Second column is the data y-vales.
        Third column the error on the data y-values.
        Fourth column the calculated y-values.
        '''
        try:
            self.data.export_data_to_files(basename)
        except data.IOError as e:
            raise IOError(e.error_message, e.file)


    def export_script(self, filename):
        '''
        Export the script to filename. Will be a python script with ASCII
        output (naturally).
        '''
        self._save_to_file(filename, self.script)

    def import_script(self, filename):
        '''import_script(self, filename) --> None

        Imports the script from file filename
        '''
        read_string = self._read_from_file(filename)
        self.set_script(read_string)
        self.compiled = False

    def import_table(self, filename):
        '''
        import the table from filename. ASCII input. tab delimited
        '''
        read_string = self._read_from_file(filename)
        self.parameters.set_ascii_input(read_string)

    def _save_to_file(self, filename, save_string):
        '''_save_to_file(self, filename, save_string) --> None

        Save the string to file with filename.
        '''
        try:
            savefile = open(filename, 'w')
        except Exception as e:
            raise IOError(e.__str__(), filename)

        # Save the string to file
        try:
            savefile.write(save_string)
        except Exception as e:
            raise IOError(e.__str__(), filename)

        savefile.close()

    def _read_from_file(self, filename):
        '''_read_from_file(self, filename) --> string

        Reads the entrie file into string and returns it.
        '''
        try:
            loadfile = open(filename, 'r')
        except Exception as e:
            raise IOError(e.__str__(), filename)

        # Read the text from file
        try:
            read_string = loadfile.read()
        except Exception as e:
            raise IOError(e.__str__(), filename)

        loadfile.close()

        return read_string


    # Get functions

    def get_parameters(self):
        '''
        get_parameters(self) --> parameters

        returns the parameters of the model. Instace of Parameters class
        '''
        return self.parameters

    def get_data(self):
        '''
        get_data(self) --> self.data

        Returns the DataList object.
        '''

        return self.data

    def get_script(self):
        '''
        get_script(self) --> self.script

        Returns the model script (string).
        '''
        return self.script

    def get_filename(self):
        '''
        get_filename(self) --> string

        returns the filename of the model id the model has not been saved
        it returns an empty string
        '''
        return self.filename

    def get_possible_parameters(self):
        ''' get_possible_parameters(self) --> objlist, funclist

        Returns a list of all the current objects that are of
        the classes defined by self.registred_classes.
        To be used in the parameter grid.
        '''
        # Start by updating the config file
        self.read_config()
        # First we should see if any of the
        # classes is defnined in model.__pars__
        # or in __pars__
        pars = []
        try:
            # Check if the have a pars in module named model
            pars = self.eval_in_model('model.__pars__')
            pars = ['model.%s'%p for p in pars]
        except:
            # Check if we have a __pars__ in the main script
            try:
                pars = self.eval_in_model('__pars__')
                pars = ['%s'%p for p in pars]
            except:
                pass
        isstrings = sum([type(p) == type('') for p in pars]) == len(pars)
        if not isstrings:
            pars = []

        # First find the classes that exists..
        # and defined in self.registred_classes
        classes=[]
        for c in self.registred_classes + pars:
            try:
                ctemp = self.eval_in_model(c)
            except:
                pass
            else:
                classes.append(ctemp)
        #Check so there are any classes defined before we proceed
        if len(classes) > 0:
            # Get all the objects in the compiled module
            names = self.script_module.__dict__.keys()
            # Create a tuple of the classes we dfined above
            tuple_of_classes = tuple(classes)
            # Creating a dictionary that holds the name of the classes
            # eaxh item for a classes is a new dictonary that holds the
            # object name and then a list of the methods.
            par_dict = {}
            [par_dict.__setitem__(clas.__name__, {}) for clas in classes]
            # find all the names of the objects that belongs to
            # one of the classes
            objs = [(name, self.eval_in_model(name)) for name in names]
            valid_objs = [(name, obj) for name, obj in objs
                          if isinstance(obj, tuple_of_classes)]
            # nested for loop for finding for each valid object
            # the right name as given by self.set_func
            # Add this to the right item in par_dict given
            # its class and name.
            [par_dict[obj.__class__.__name__].__setitem__(name,
                     [member for member in dir(obj)
                        if member[:len(self.set_func)] == self.set_func])
             for name,obj in valid_objs]

            return par_dict

        return {}


    # Set functions - a necessary evil...

    def set_script(self, text):
        '''
        Set the text in the script use this to change the model script.
        '''
        self.script = text

    def set_fom_func(self, fom_func):
        '''
        Set the fucntion that calculates the figure of merit between the model
        and the data.
        '''
        if type(fom_func)==str:
            self.fom_func = getattr(fom_funcs,fom_func)
        else:
            self.fom_func = fom_func

    def set_weighting_factor(self,factor):
        # fom_funcs.weight_pars.weight_factor = factor
        #fom_funcs.weight_pars_instance.set_weight_factor(factor)
        self.weight_factor = factor

    def set_weighting_region(self,region):
        # fom_funcs.weight_pars.weight_map = region
        #fom_funcs.weight_pars_instance.set_weight_map(region)
        self.weight_map = region

    def is_compiled(self):
        '''is_compiled(self) --> compiled [boolean]

        Returns true if the model script has been sucessfully
        compiled.
        '''
        return self.compiled

#END: Class Model
#==============================================================================
#Some Exception definition for errorpassing
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ParameterError(GenericError):
    ''' Class for yielding Parameter errors
    '''
    def __init__(self, parameter, parameter_number, error_message, what = -1):
        '''__init__(self, parameter, parameter_number, error_message) --> None

        parameter: the name of the parameter [string]
        parameter_number: the position of the parameter in the list [int]
        error_mesage: pythons error message from the original exception
        set: int to show where the error lies.
            -1 : undefined
             0 : an not find the parameter
             1 : can not evaluate i.e. set the parameter
             2 : value are larger than max
             3 : value are smaller than min
             4 : No parameters to fit
        '''
        self.parameter = parameter
        self.parameter_number = parameter_number
        self.error_message = error_message
        self.what = what

    def __str__(self):
        ''' __str__(self) --> text [string]
        Yields a human readable description of the problem
        '''
        text = ''
        text += 'Parameter number %i, %s, '%(self.parameter_number+1,\
            self.parameter)

        # Take care of the different cases
        if self.what == 0:
            text += 'could not be found. Check the spelling.\n'
        elif self.what == 1:
            text += 'could not be evaluated. Check the code of the function.\n'
        elif self.what == 2:
            text += 'is larger than the value in the max column.\n'
        elif self.what == 3:
            text += 'is smaller than the value in the min column\n'
        elif self.what == 4:
            text = 'There are no parameter selcted to be fitted.\n' + \
                    'Select the parameters you want to fit by checking the ' +\
                    'boxes in the fit column, folder grid'
        else:
            text += 'yielded an undefined error. Check the Python output\n'

        if self.error_message != 'None':
            text += '\nPython error output:\n' + self.error_message

        return text

class ModelError(GenericError):
    ''' Class for yielding compile or evaluation errors in the model text
    '''
    def __init__(self, error_message, where):
        '''__init__(self, error_message, where = -1) --> None

        error_mesage: pythons error message from the original exception
        where: integer describing where the error was raised.
                -1: undef
                 0: compile error
                 1: evaulation error
        '''
        self.error_message = error_message
        self.where = where

    def __str__(self):
        ''' __str__(self) --> text [string]
        Yields a human readable description of the problem
        '''
        text = ''
        if self.where == 0:
            text += 'It was not possible to compile the model script.\n'
        elif self.where == 1:
            text += 'It was not possible to evaluate the model script.\n'\
                    + 'Check the Sim function.\n'
        elif self.where == -1:
            text += 'Undefined error from the Model. See below.\n'

        text += '\n' + self.error_message

        return text

class FomError(GenericError):
    '''Error class for the fom evaluation'''
    def __init__(self, error_message):
        ''' __init__(self, error_message) --> None'''
        self.error_message = error_message

    def __str__(self):
        text = 'Could not evaluate the FOM function. See python output.\n'\
            + '\n' + self.error_message
        return text

class IOError(GenericError):
    ''' Error class for input output, mostly concerning files'''

    def __init__(self, error_message, file = ''):
        '''__init__(self, error_message)'''
        self.error_message = error_message
        self.file = file

    def __str__(self):
        text = 'Input/Output error for file:\n' + self.file +\
                '\n\n Python error:\n ' + self.error_message
        return text


# Some small default function that are needed for initilization

def default_fom_func(simulated_data, data):
    '''
    The default fom function. Its just a dummy so far dont use it!
    '''
    return sum([abs(d.y-sim_d).sum() for sim_d, d \
                in zip(simulated_data,data)])



class Temp:
    pass
