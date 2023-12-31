'''
Library for the classes to store the data. The class DataSet stores
on set and the class DataList stores multiple DataSets.
Programmer Matts Bjorck
Last changed: 2008 08 22
'''

from numpy import *
import numpy as np
import os, time

#==============================================================================
#BEGIN: Class DataSet

class DataSet:
    ''' Class to store each dataset to fit. To fit several items instead the.
        Contains x,y,error values and xraw,yraw,errorraw for the data.
    '''
    def __init__(self, name = '', copy_from = None):
        #Processed data
        self.x = array([])
        self.y = array([])
        self.y_sim = array([])
        self.y_fom = array([])
        self.error = array([])
        # The raw data
        self.x_raw = array([])
        self.y_raw = array([])
        self.error_raw = array([])

        self.extra_data = {}
        self.extra_data_raw = {}
        # This is to add datasets that can be oprated upon as x,y and z
        self.extra_commands = {}

        # The different commands to transform raw data to normal data
        self.x_command = 'x'
        self.y_command = 'y'
        self.error_command = 'e'

        # Should we display the dataset, ie plot it
        # This should be default for ALL datasets..
        self.show = True

        # Special list for settings when setting the plotting properties
        self.plot_setting_names = ['color', 'symbol', 'symbolsize', 'linetype',\
                              'linethickness']
        # Name of the data set
        if name == '':
            self.name = 'New Data'
        else:
            self.name = name

        if copy_from:
            #Should the dataset be used for fitting?
            self.use = copy_from.use
            #Should the error be used
            self.use_error = copy_from.use_error
            #The columns to load
            self.cols = copy_from.cols # Columns to load (xcol,ycol,ecol)
            #The different colors for the data and simulation
            self.data_color = copy_from.data_color
            self.sim_color = copy_from.sim_color
            # The different linetypes and symbols incl. sizes
            self.data_symbol = copy_from.data_symbol
            self.data_symbolsize = copy_from.data_symbolsize
            self.data_linetype = copy_from.data_linetype
            self.data_linethickness = copy_from.data_linethickness
            self.sim_symbol = copy_from.sim_symbol
            self.sim_symbolsize = copy_from.sim_symbolsize
            self.sim_linetype = copy_from.sim_linetype
            self.sim_linethickness = copy_from.sim_linethickness
        else:
            #Should the dataset be used for fitting?
            self.use = True
            #Should the error be used
            self.use_error = False
            #The columns to load
            self.cols = [0,1,1] # Columns to load (xcol,ycol)
            #The different colors for the data and simulation
            self.data_color = (0.0, 0.0, 1.0)
            self.sim_color = (1.0, 0.0, 0.0)
            # The different linetypes and symbols incl. sizes
            self.data_symbol = 'o'
            self.data_symbolsize = 4
            self.data_linetype = '-'
            self.data_linethickness = 2
            self.sim_symbol = ''
            self.sim_symbolsize = 1
            self.sim_linetype = '-'
            self.sim_linethickness = 2

    def copy(self):
        ''' Make a copy of the current Data Set'''
        cpy = DataSet()
        cpy.safe_copy(self)
        return cpy

    def apply_mask(self):
        assert hasattr(self,'mask') and (len(self.mask) == len(self.x)),'dimension not matched!'
        self.x = self.x[self.mask]
        self.y = self.y[self.mask]
        self.error = self.error[self.mask]
        self.x_raw = self.x_raw[self.mask]
        self.y_raw = self.y_raw[self.mask]
        self.error_raw = self.error_raw[self.mask]
        for each in self.extra_data:
            self.extra_data[each] = self.extra_data[each][self.mask]
        for each in self.extra_data_raw:
            self.extra_data_raw[each] = self.extra_data_raw[each][self.mask]
        self.mask = self.mask[self.mask]

    def safe_copy(self, new_set):
        '''safe_copy(self, new_set) --> None

        A safe copy from one dataset to another.
        Note, not totally safe since references are not broken
        '''
        self.name = new_set.name
        self.x = new_set.x
        self.y = new_set.y
        self.y_sim = new_set.y_sim
        self.error = new_set.error
        # The raw data
        self.x_raw = new_set.x_raw
        self.y_raw = new_set.y_raw
        self.error_raw = new_set.error_raw
        try:
            self.mask = new_set.mask
        except:
            pass

        # The dictonaries for the extra data
        try:
            self.extra_data = new_set.extra_data
        except AttributeError:
            self.extra_data = {}
        try:
            self.extra_data_raw = new_set.extra_raw
        except AttributeError:
            self.extra_data_raw = self.extra_data.copy()
        try:
            self.extra_commands = new_set.extra_commands
        except AttributeError:
            self.extra_commands = {}

        # The different commands to transform raw data to normal data
        self.x_command = new_set.x_command
        self.y_command = new_set.y_command
        self.error_command = new_set.error_command
        try:
            self.show = new_set.show
        except AttributeError:
            self.show = True

        self.use = new_set.use
        #Should the error be used
        self.use_error = new_set.use_error
        #The columns to load
        #The different colors for the data and simulation
        self.data_color = new_set.data_color
        self.sim_color = new_set.sim_color
        # The different linetypes and symbols incl. sizes
        self.data_symbol = new_set.data_symbol
        self.data_symbolsize = new_set.data_symbolsize
        self.data_linetype = new_set.data_linetype
        self.data_linethickness = new_set.data_linethickness
        self.sim_symbol = new_set.sim_symbol
        self.sim_symbolsize = new_set.sim_symbolsize
        self.sim_linetype = new_set.sim_linetype
        self.sim_linethickness = new_set.sim_linethickness

    def copy_and_extend(self, dL=0.08, pt=10):
        ''' Make a copy of the current Data Set'''
        cpy = DataSet()
        cpy.safe_copy(self)
        cpy.safe_copy_and_extend_l(self, dL, pt)
        return cpy

    #extend l to Bragg_L + or - dL, the extended l segment has pt points
    #it will only work for simply situation where there is only one dL
    #i.e. the Bragg's peaks are evently spaced, a more complicated func will 
    #needed if other than this situation
    def safe_copy_and_extend_l(self, new_set, dL=0.08, pt=10):
        inserted_l = []
        l = self.x
        dl = list(set(self.extra_data['dL']))[0]
        lB = list(set(self.extra_data['LB']))[0]
        Bragg_Peaks = [lB+i*dl for i in range(10) if min(l)<lB+i*dl<max(l)]
        for each in Bragg_Peaks:
            index_ = np.argmin(abs(np.array(l)-each))
            if l[index_]<each:
                index_lf, index_rt = index_, int(index_+1)
            else:
                index_lf, index_rt = int(index_-1), index_
            inserted_l = inserted_l + np.arange(l[index_lf]+dL/pt, each-0.02+dL/pt, dL/pt).tolist()
            inserted_l = inserted_l + np.arange(each+0.02, l[index_rt], dL/pt).tolist()
        
        num_ = len(inserted_l)
        self.x = np.array(new_set.x.tolist()+inserted_l)
        self.y = np.array(new_set.y.tolist()+[0]*num_)
        self.y_sim = np.array(new_set.y_sim.tolist() +[0]*num_)
        self.error = np.array(new_set.error.tolist()+[0]*num_)
        # The raw data
        self.x_raw = np.array(new_set.x_raw.tolist()+inserted_l)
        self.y_raw = np.array(new_set.y_raw.tolist()+[0]*num_)
        self.error_raw = np.array(new_set.error_raw.tolist()+[0]*num_)
        self.mask = np.array(new_set.mask.tolist() + [True]*num_)
        for each in self.extra_data.keys():
            self.extra_data[each ]= np.array(self.extra_data[each].tolist()+[self.extra_data[each][0]]*num_)

    def get_extra_data_names(self):
        '''get_extra_data_names(self) --> names [list]

        returns the names of the extra data
        '''
        return self.extra_data.keys()

    def set_extra_data(self, name, value, command = None):
        '''set_extra_data_names(self, name, value, command = None)

        sets extra data name, if it does not exist a new entry is created.
        name should be a string and value can be any object.
        If command is set, this means that the data set can be operated upon
        with commands just as the x,y and e data members.
        '''
        if name in ['x', 'y', 'e']:
            raise KeyError('The extra data can not support the key'
                           'names x, y or e.')
        self.extra_data[name] = value
        self.extra_data_raw[name] = value
        if command:
            self.extra_commands[name] = command

    def get_extra_data(self, name):
        '''get_extra_data(self, name) --> object

        returns the extra_data object with name name [string] if does not
        exist an LookupError is yielded.
        '''
        if not (name in self.extra_data):
            raise LookupError('Can not find extra data with name %s'%name)

        return self.extra_data[name]

    def loadfile(self,filename, sep='\t', pos=0):
        '''
        Function to load data from a file.
        Note that the data should be in ASCII format and it can
        be gzipped (handled automagically if the filname ends with .gz
        Possible extras:
        comments - string of chars that shows that a line is a comment
        delimeter - chars that are spacers between values default None
            all whitespaces
        skiprows - number of rows to skip before starting to read the data

        '''
        try:
            f=open(filename)
            #f.close()
        except:
            print("Can't open file: %s"%filename)
        else:
            try:
                A = loadtxt(f)
                #, comments = '#', delimeter = None, skiprows = 0
            except:
                print("Can't read the file %s, check the format"%filename)
            else:
                #print A
                xcol=self.cols[0]
                ycol=self.cols[1]
                ecol=self.cols[2]
                #print xcol,ycol
                if xcol<A.shape[1] and ycol<A.shape[1] and ecol<A.shape[1]:
                    self.x_raw=A[:,xcol].copy()
                    self.y_raw=A[:,ycol].copy()
                    self.error_raw=A[:,ecol].copy()
                    self.x=A[:,xcol]
                    self.y=A[:,ycol]
                    self.error=A[:,ecol]
                    self.y_sim = array([])
                    print("Sucessfully loaded %i datapoints"%(A.shape[0]))
                    return True
                else:
                    print("There are not enough columns in your data\n\
                     As I see it there are %i columns"%A.shape[1])
            return False

    def loadfile_new(self,filename, xye_col=[2,3,4],extra_col={'h':0,'k':1,'LB':5,'dL':6}):
        '''
        compared to the original loadfile, here we also load extra columns
        xye_col=[2,3,4]:x on second column, y on third and so so
        extra_col={'h':0,'k':1,'LB':5,'dL':6}: h on 0th column, k on first column and so on
        name for the dataset is simply (h,k) (eg (1,0),(0,0))
        '''
        try:
            f=open(filename)
            #f.close()
        except:
            print("Can't open file: %s"%filename)
        else:
            try:
                A = loadtxt(f)
                #, comments = '#', delimeter = None, skiprows = 0
            except:
                print("Can't read the file %s, check the format"%filename)
            else:
                #print A
                self.cols[0],xcol=xye_col[0],xye_col[0]
                self.cols[1],ycol=xye_col[1],xye_col[1]
                self.cols[2],ecol=xye_col[2],xye_col[2]
                #print xcol,ycol
                if xcol<A.shape[1] and ycol<A.shape[1] and ecol<A.shape[1]:


                    self.name="(%i,%i)"%(A[0,0],A[0,1])
                    self.x_raw=A[:,xcol].copy()
                    self.y_raw=A[:,ycol].copy()
                    self.error_raw=A[:,ecol].copy()
                    self.x=A[:,xcol]
                    self.y=A[:,ycol]
                    self.error=A[:,ecol]
                    self.y_sim = array([])
                    for key in extra_col.keys():
                        self.extra_data[key]=A[:,extra_col[key]].copy()
                        self.extra_data_raw[key]=A[:,extra_col[key]].copy()
                    print("Sucessfully loaded %i datapoints"%(A.shape[0]))
                    return True
                else:
                    print("There are not enough columns in your data\n\
                     As I see it there are %i columns"%A.shape[1])
            return False

    def save_file(self, filename):
        '''save_file(self, filename) --> None

        saves the dataset to a file with filename.
        '''
        if  self.x.shape == self.y_sim.shape and \
            self.y.shape == self.error.shape and\
            self.x.shape == self.y.shape:
            # save the file
            #print self.y.shape, self.y_sim.shape
            #print c_[self.x, self.y_sim, self.y, self.error]
            f = open(filename, 'w')
            f.write('# Dataset "%s" exported from GenX on %s\n'%\
                            (self.name, time.ctime()))
            f.write('# Column lables:\n')
            f.write('# x\tI_simulated\tI\terror(I)\n')
            savetxt(f, c_[self.x, self.y_sim, self.y, self.error])
        else:
            debug = 'y_sim.shape: ' + str(self.y_sim.shape) + '\ny.shape: ' +\
            str(self.y.shape) + '\nx.shape: ' + str(self.x.shape) +\
            '\nerror.shape: ' + str(self.error.shape)
            #print debug
            raise IOError('The data is not in the correct format all the' +\
                    'arrays have to have the same shape:\n' + debug, filename)


    def run_x_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw

        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        self.x = eval(self.x_command)
        #print self.x

    def run_y_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw

        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        self.y = eval(self.y_command)
        #print self.y
        #print self.y_command

    def run_error_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw

        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        self.error = eval(self.error_command)

    def run_extra_commands(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw

        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        for key in self.extra_commands:
            exec('self.extra_data["%s"] = eval(self.extra_commands["%s"])'\
                    %(key, key))


    def run_command(self):
        self.run_x_command()
        self.run_y_command()
        self.run_error_command()
        self.run_extra_commands()

    def try_commands(self, command_dict):
        ''' try_commands(self, command_dict) --> tuple of bool
        Evals the commands to locate any errors. Used to
        test the commands before doing the actual setting of x,y and z
        '''
        result = ''

        x = self.x_raw
        y = self.y_raw
        e = self.error_raw

        #Know we have to do this with the extra data
        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        xt = self.x
        yt = self.y
        et = self.error

        #Know we have to do this with the extra data
        for key in self.extra_data_raw:
            exec('%st = self.extra_data["%s"]'%(key, key))

        # Try to evaluate all the expressions
        if command_dict['x'] != '':
            try:
                xt = eval(command_dict['x'])
            except Exception as e:
                result += 'Error in evaluating x expression.\n\nPython output:\n'\
                            + e.__str__() + '\n'

        if command_dict['y'] != '':
            try:
                yt = eval(command_dict['y'])
            except Exception as e:
                result += 'Error in evaluating y expression.\n\nPython output:\n'\
                        + e.__str__() + '\n'

        if command_dict['e'] != '':
            try:
                et = eval(command_dict['e'])
            except Exception as e:
                result += 'Error in evaluating e expression.\n\nPython output:\n'\
                        + e.__str__() + '\n'

        for key in self.extra_commands:
            if command_dict[key] != '':
                try:
                    exec('%st = eval(command_dict["%s"])'%(key, key))
                except Exception as e:
                    result += 'Error in evaluating %s expression.\n\nPython output:\n'%key\
                            + e.__str__() + '\n'

        # If we got an error - report it
        if result != '':
            return result
        #print 'Debug, datatry: ', xt, yt, et
        # Finally check so that all the arrays have the same size
        extra_shape = not all([eval('%st.shape'%key) == xt.shape for \
                            key in self.extra_commands])
        if (xt.shape != yt.shape or xt.shape != et.shape or extra_shape)\
            and result == '':
            result += 'The resulting arrays are not of the same size:\n' + \
                       'len(x) = %d, len(y) = %d, len(e) = %d'\
                    %(xt.shape[0], yt.shape[0], et.shape[0])
            for key in self.extra_commands:
                result += ', len(%s) = %d'%(key, eval('%st.shape[0]'%key))
        return result

    def get_commands(self):
        ''' get_commands(self) --> list of dicts
        Returns the commnds as a dictonary with items x, y, z
        '''
        cmds = {'x':self.x_command, 'y':self.y_command, 'e':self.error_command}
        for key in self.extra_commands:
            cmds[key] = self.extra_commands[key]
        return cmds

    def set_commands(self, command_dict):
        ''' set_commands(self, command_dict) --> None
        Sets the commands in the data accroding to values in command dict
        See get_commands for more details
        '''
        if command_dict['x'] != '':
            self.x_command = command_dict['x']
        if command_dict['y'] != '':
            self.y_command = command_dict['y']
        if command_dict['e'] != '':
            self.error_command = command_dict['e']
        # Lets do it for the extra commands as well
        for key in command_dict:
            if self.extra_commands.has_key(key):
                if command_dict[key] != '':
                    self.extra_commands[key] = command_dict[key]

    def set_simulated_data(self, simulated_data):
        self.y_sim = simulated_data

    def set_fom_data(self, fom_data):
        self.y_fom = fom_data

    def get_sim_plot_items(self):
        '''get_sim_plot_items(self) --> dict
        Returns a dictonary of color [tuple], symbol [string],
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the simulation.
        '''
        return {'color': (self.sim_color[0]*255, self.sim_color[1]*255,\
     self.sim_color[2]*255),\
                'symbol': self.sim_symbol,\
                'symbolsize': self.sim_symbolsize,\
                'linetype': self.sim_linetype,\
                'linethickness': self.sim_linethickness\
               }
    def get_data_plot_items(self):
        '''get_data_plot_items(self) --> dict
        Returns a dictonary of color [tuple], symbol [string],
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the data.
        '''
        return {'color': (self.data_color[0]*255, self.data_color[1]*255,\
     self.data_color[2]*255),\
                'symbol': self.data_symbol,\
                'symbolsize': self.data_symbolsize,\
                'linetype': self.data_linetype,\
                'linethickness': self.data_linethickness\
               }

    def set_data_plot_items(self, pars):
        ''' set_data_plot_items(self, pars) --> None
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        #print 'data set_data_plot_items: '
        #print pars
        for name in self.plot_setting_names:
            if pars[name] != None:
                if type(pars[name]) == type(''):
                    exec('self.data_' + name + ' = "' \
                            + pars[name].__str__() + '"')
                elif name == 'color':
                    c = pars['color']
                    self.data_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.data_' + name + ' = ' + pars[name].__str__())

    def set_sim_plot_items(self, pars):
        ''' set_data_plot_items(self, pars) --> None
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        #print 'data set_sim_plot_items: '
        #print pars
        for name in self.plot_setting_names:
            if pars[name] != None:
                if type(pars[name]) == type(''):
                    exec('self.sim_' + name + ' = "' \
                            + pars[name].__str__() + '"')
                elif name == 'color':
                    c = pars['color']
                    self.sim_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.sim_' + name + ' = ' + pars[name].__str__())

    def set_show(self, val):
        '''Set show true - show data set in plots
        '''
        self.show = bool(val)


#END: Class DataSet
#==============================================================================
#BEGIN: Class DataList
class DataList:
    ''' Class to store a list of DataSets'''

    def __init__(self):
        ''' init function - creates a list with one DataSet'''
        self.items=[DataSet(name='Data 0')]
        self._counter=1
        self.ctr_data_all = None#numpy array to concatenate all datasets, columns =[h, k, x, y, LD, dL, mask]
        self.ctr_data_info = {}#store info of the datasets

    def concatenate_all_ctr_datasets(self):
        all_ctr_data = []
        self.ctr_data_info = {}
        self.scaling_tag = []
        self.scaling_tag_raxs = []
        self.data_sequence = []
        current_raxs_tag = 100#first raxs dataset start from 100
        for i,each in enumerate(self.items):
            if hasattr(each,'mask'):
                h,k,x,y,LB,dL = each.extra_data['h'][each.mask][:,np.newaxis],each.extra_data['k'][each.mask][:,np.newaxis],each.x[each.mask][:,np.newaxis],each.extra_data['Y'][each.mask][:,np.newaxis],each.extra_data['LB'][each.mask][:,np.newaxis],each.extra_data['dL'][each.mask][:,np.newaxis]
            else:
                h,k,x,y,LB,dL = each.extra_data['h'][:,np.newaxis],each.extra_data['k'][:,np.newaxis],each.x[:,np.newaxis],each.extra_data['Y'][:,np.newaxis],each.extra_data['LB'][:,np.newaxis],each.extra_data['dL'][:,np.newaxis]

            if x[0]>100:#x column is energy for raxs (in ev); but x column is L for CTR (smaller than 10 usually)
                #data_type_tag = 100+i#datasets with tag>100 are raxs data 
                data_type_tag = current_raxs_tag
                current_raxs_tag +=  1
                self.data_sequence.append(data_type_tag)
                if int(h[0])==0 and int(k[0])==0:
                    self.scaling_tag_raxs.append('specular_rod')
                else:
                    self.scaling_tag_raxs.append('nonspecular_rod')
            else:
                data_type_tag = 1+i#datasets with tag>1 but <100 are ctr data
                self.data_sequence.append(data_type_tag)
                if int(h[0])==0 and int(k[0])==0:
                    self.scaling_tag.append('specular_rod')
                else:
                    self.scaling_tag.append('nonspecular_rod')
            self.ctr_data_info[data_type_tag] = len(h)
            #mask = np.ones(len(h))[:,np.newaxis]
            fbulk = np.zeros(len(h))[:,np.newaxis]
            temp_data = np.hstack((h,k,x,y,LB,dL,fbulk,np.zeros(len(h))[:,np.newaxis] + data_type_tag))
            if len(all_ctr_data)==0:
                all_ctr_data = temp_data
            else:
                all_ctr_data = np.vstack((all_ctr_data,temp_data))
        self.ctr_data_all = all_ctr_data
        self.ctr_data_summary = {each:item for each, item in self.ctr_data_info.items() if each<100}
        self.raxs_data_summary = {each:item for each, item in self.ctr_data_info.items() if each>=100}
        return all_ctr_data

    def binary_comparison_and(self, bool_list1=[True, False,True], bool_list2=[False,False,True]):
        return [int(first)+int(second) == 2 for first, second in zip(bool_list1, bool_list2)]

    def save_full_dataset(self, filename):
        np.savetxt(filename, self.ctr_data_all[:,[0,1,2,3,4,5]],header = '#h k x y LB dL')

    def split_fullset(self,full_set,scale_factors, data_type = 'CTR'):
        if data_type == 'CTR':
            #datasets with tag>1 but <100 are ctr data
            data_info = {each:item for each, item in self.ctr_data_info.items() if each<100}
        elif data_type == 'RAXS':
            #datasets with tag>100 are raxs data
            data_info = {each:item for each, item in self.ctr_data_info.items() if each>=100}
        sub_sets = []
        cum_sum = np.cumsum([0]+list(data_info.values()))
        if type(scale_factors)!=type([]):
            scale_factors = [scale_factors]*len(data_info)
        else:
            assert len(scale_factors) == len(data_info),'The length of scale_factors and total number of ctr datasets do not match each other!'
        for i in range(len(data_info)):
            sub_sets.append(full_set[cum_sum[i]:cum_sum[i+1]]*scale_factors[i])
        return sub_sets

    def split_used_dataset(self,full_set, data_type = 'CTR'):
        if data_type == 'CTR':
            #datasets with tag>1 but <100 are ctr data
            data_info = {each:item for each, item in self.ctr_data_info.items() if (each<100 and self.items[self.data_sequence.index(each)].use)}
        elif data_type == 'RAXS':
            #datasets with tag>100 are raxs data
            data_info = {each:item for each, item in self.ctr_data_info.items() if (each>=100 and self.items[self.data_sequence.index(each)].use)}
        sub_sets = []
        cum_sum = np.cumsum([0]+list(data_info.values()))
        for i in range(len(data_info)):
            sub_sets.append(full_set[cum_sum[i]:cum_sum[i+1]])
        return sub_sets, data_info

    #update the partial datasets to the full_set
    #full_set a list
    #sub_sets a list of list, each item is one dataset
    #data_info is a dict giving the info about the sub_sets, ie key = dataset_id, value = length of the associated dataset
    #data_type = 'CTR' or 'RAXS'
    def insert_datasets(self, full_set, sub_sets, data_info, data_type = 'CTR'):
        if data_type == 'CTR':
            keys_all = list(self.ctr_data_summary.keys())
            begin_indexs = []
            for each in data_info:
                begin_indexs.append(sum([self.ctr_data_summary[keys_all[i]] for i in range(keys_all.index(each))]))
            for i in range(len(begin_indexs)):
                end_index = int(begin_indexs[i]+data_info[list(data_info.keys())[i]])
                full_set[int(begin_indexs[i]):end_index] = sub_sets[i]
        elif data_type == 'RAXS':
            keys_all = list(self.raxs_data_summary.keys())
            begin_indexs = []
            for each in data_info:
                begin_indexs.append(sum([self.raxs_data_summary[keys_all[i]] for i in range(keys_all.index(each))]))
            for i in range(len(begin_indexs)):
                end_index = int(begin_indexs[i]+data_info[list(data_info.keys())[i]])
                full_set[int(begin_indexs[i]):end_index] = sub_sets[i]
        return full_set

    def merge_datasets(self, ctr_datasets, raxs_datasets):
        assert (len(ctr_datasets)+len(raxs_datasets))==len(self.ctr_data_info),'The length of datasets does not match the total length of provided datasets!'
        full_sets = []
        begin_index_ctr = 0
        begin_index_raxs = 0
        for each in self.ctr_data_info:
            if each<100:
                full_sets.append(ctr_datasets[begin_index_ctr])
                begin_index_ctr = begin_index_ctr + 1
            elif each>=100:
                full_sets.append(raxs_datasets[begin_index_raxs])
                begin_index_raxs = begin_index_raxs + 1
        return full_sets

    def __getitem__(self,key):
        '''__getitem__(self,key) --> DataSet

        returns item at position key
        '''
        return self.items[key]

    def __iter__(self):
        ''' __iter__(self) --> iterator

        Opertor definition. Good to have in case one needs to loop over
        all datasets
        '''
        return self.items.__iter__()

    def __len__(self):
        '''__len__(self) --> length (integer)

        Returns the nmber of datasers in the list.
        '''
        return self.items.__len__()

    def safe_copy(self, new_data):
        '''safe_copy(self, new_data) --> None

        Conduct a safe copy of a data set into this data set.
        This is intended to produce version safe import of data sets.
        '''
        self.items = []
        for new_set in new_data:
            self.items.append(DataSet())
            self.items[-1].safe_copy(new_set)

    def add_new(self,name=''):
        ''' add_new(self,name='') --> None

        Adds a new DataSet with the optional name. If name not sets it
        will be given an automatic name
        '''
        if name=='':
            self.items.append(DataSet('Data %d'%self._counter,\
                        copy_from=self.items[-1]))
            self._counter+=1
        else:
            self.items.append(DataSet(name,copy_from=self.items[-1]))
        # self.concatenate_all_ctr_datasets()
        #print "An empty dataset is appended at postition %i."%(len(self.items)-1)

    def add_new_list(self,name_list=['']):
        '''
        Adds a list of DataSet, name_list is a list of dataset, each item is the absolute path to one dataset
        '''
        for name in name_list:
            self.add_new('')
            self.items[-1].loadfile_new(name)
        self.items=self.items[1:]
        self._counter=self._counter-1
        self.concatenate_all_ctr_datasets()
        #print "An empty dataset is appended at postition %i."%(len(self.items)-1)

    def delete_item(self,pos):
        '''delete_item(self,pos) --> None

        Deletes the item at position pos. Only deletes if the pos is an
        element and the number of datasets are more than one.
        '''
        if pos<len(self.items) and len(self.items)>1:
            self.items.pop(pos)
            print("Data set number %i have been removed."%pos)
            self.concatenate_all_ctr_datasets()
            return True
        else:
            print('Can not remove dataset number %i.'%pos)
            return False

    def move_up(self, pos):
        '''move_up(self, pos) --> None

        Move the data set at position pos up one step. If it is at the top
        it will not be moved.
        '''
        if pos != 0:
            tmp = self.items.pop(pos)
            self.items.insert(pos-1, tmp)

    def move_down(self,pos):
        '''
        move_down(self,pos) --> None

        Move the dataset at postion pos down one step. If it is at the bottom
        it will not be moved.
        '''
        if pos != len(self.items):
            tmp = self.items.pop(pos)
            self.items.insert(pos + 1, tmp)

    def update_data(self):
        ''' update_data(self) --> None

        Calcultes all the values for the current items.
        '''
        [item.run_command() for item in self.items]

    def set_simulated_data(self, sim_data):
        '''
        set_simulated_data(self, sim_data) --> None

        Sets the simualted data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_simulated_data(sim_data[i]) for i in\
            range(self.get_len())]

    def set_fom_data(self, fom_data):
        '''
        set_fom_data(self, fom_data) --> None

        Sets the point by point fom data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_fom_data(fom_data[i]) for i in\
            range(self.get_len())]

    def get_len(self):
        return len(self.items)

    def get_name(self,pos):
        '''
        get_name(self,pos) --> name (string)

        Yields the name(string) of the dataset at position pos(int).
        '''
        return self.items[pos].name

    def get_cols(self,pos):
        return self.items[pos].cols

    def get_use(self, pos):
        '''get_use_error(self, pos) --> bool
        returns the flag use for dataset at pos [int].
        '''
        return self.items[pos].use

    def get_use_error(self, pos):
        '''get_use_error(self, pos) --> bool
        returns the flag use_error for dataset at pos [int].
        '''
        return self.items[pos].use_error

    def toggle_use_error(self, pos):
        '''toggle_use_error(self, pos) --> None
        Toggles the use_error flag for dataset at position pos.
        '''
        self.items[pos].use_error = not self.items[pos].use_error

    def toggle_use(self, pos):
        '''toggle_use(self, pos) --> None
        Toggles the use flag for dataset at position pos.
        '''
        self.items[pos].use = not self.items[pos].use

    def toggle_show(self, pos):
        '''toggle_show(self, pos) --> None
        Toggles the show flag for dataset at position pos.
        '''
        self.items[pos].show = not self.items[pos].show

    def show_items(self, positions):
        '''show_items(self, positions) --> None
        Will put the datasets at positions [list] to show all
        other of no show, hide.
        '''
        [item.set_show(i in positions) for i, item in enumerate(self.items)]


    def set_name(self,pos,name):
        '''
        set_name(self,pos,name) --> None

        Sets the name of the data set at position pos (int) to name (string)
        '''
        self.items[pos].name=name

    def export_data_to_files(self, basename, indices = None):
        '''export_data_to_files(self, basename, indices = None) --> None

        saves the data to files with base name basename and extentions .dat
        If indices are used only the data given in the list indices are
        exported.
        '''
        # Check if we shoudlstart picking data sets to export

        if indices:
            if not sum([i < len(self.items) for i in indices]) == len(indices):
                raise 'Error in export_data_to_files'
        else:
            indices = range(len(self.items))
        #print 'Output: ', indices, len(self.items)
        for index in indices:
            base, ext = os.path.splitext(basename)
            if ext == '':
                ext = '.dat'
            self.items[index].save_file(base + '%03d'%index + ext)


    def get_data_as_asciitable(self, indices = None):
        ''' get_data_as_table(self, indices = None) --> string

        Yields the data sets as a ascii table with tab seperated values.
        This makes it possible to export the data to for example spreadsheets.
        Each data set will be four columns with x, Meas, Meas error and Calc.
        If none is given all the data sets are transformed otherwise incdices
        shouldbe a list.
        '''

        if indices:
            if not sum([i < len(self.items) for i in indices]) == len(indices):
                raise 'Error in get_data_as_asciitable'
        else:
            indices = range(len(self.items))

        #making some nice looking header so the user know what is what
        header1=''.join(['%s\t\t\t\t'%self.items[index].name\
                            for index in indices])
        header2=''.join(['x\ty\ty error\ty sim\t' for index in indices])

        # Find the maximum extent of the data sets
        maxlen=max([len(item.y_sim) for item in self.items])

        # Create the funtion that actually do the exporting
        def exportFunc(index,row):
            item = self.items[index]
            if row < len(item.x):
                return '%e\t%e\t%e\t%e\t'%(item.x[row], item.y[row],\
                                            item.error[row], item.y_sim[row])
            else:
                return ' \t \t \t \t'
        # Now create the data
        text_data = ''.join(['\n' + ''.join([exportFunc(index,row)\
                        for index in indices])\
                        for row in range(maxlen)])
        return header1 + '\n' + header2 + text_data

#==============================================================================
#Some Exception definition for errorpassing
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

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
