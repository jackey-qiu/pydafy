import numpy as np 
import os
import copy 
from dafy.core.util.path import DaFy_path
from dafy.projects.superrod.core.models import solvergui
from dafy.core.util.fom_funcs import *
from dafy.projects.superrod.core.models import model

#uncomment the following lines if used with slurm bash script
'''
import ray
ray.shutdown()
diffev._cpu_count = int(sys.argv[2])
redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
'''

#provide the folder where all model files (*.rod) are stored
folder_holding_model_files = os.path.join(DaFy_path,"examples/hematite_rcut_AD/test_batch")
#folder_holding_model_files = "/Users/canrong/apps/DaFy/examples/Cu100_CO2_EC/test_batch"

def obtain_rod_files(folder):
    '''
    load all rod files(*.rod) located in a selected folder
    '''
    files = []
    for file in os.listdir(folder):
        if file.endswith('.rod'):
            files.append(os.path.join(folder,file))
    return files

def get_data_type_tag(model):
    condition_raxs = model.data.ctr_data_all[:,-1]>=100
    return list(set(model.data.ctr_data_all[condition_raxs][:,-1])),list(set(model.data.ctr_data_all[:,-1]))

def set_model(model, raxs_index, raxs_index_in_all_datasets):
    for i in range(len(model.data)):
        if i!=raxs_index_in_all_datasets:
            model.data[i].use = False
        else:
            model.data[i].use = True
    for i in range(len(model.parameters.data)):
        if model.parameters.data[i][0] in [f"rgh_raxs.setA_{raxs_index+1}",f"rgh_raxs.setP_{raxs_index+1}",f"rgh_raxs.setA{raxs_index+1}",f"rgh_raxs.setB{raxs_index+1}",f"rgh_raxs.setC{raxs_index+1}"]:
            model.parameters.data[i][2] = True
        else:
            model.parameters.data[i][2] = False
    return model

#partial set, add as many as you want
#key is the set funcs defined in /.../DaFy/EnginePool/diffev.py
#values are the associated value to be set
solver_settings = {
                   "set_pop_mult":False,
                   "set_pop_size":100,
                   "set_max_generations":1000,
                   "set_autosave_interval":200
                  }

RAXS_FIT = False
model = model.Model()
solver = solvergui.SolverController(model)

for each_file in obtain_rod_files(folder_holding_model_files):
    print(f"Loading file:{each_file}")
    model.load(each_file)
    model.apply_addition_to_optimizer(solver.optimizer)
    #set mask points
    for each in model.data_original:
        if not hasattr(each,'mask'):
            each.mask = np.array([True]*len(each.x))
    for each in model.data:
        if not hasattr(each,'mask'):
            each.mask = np.array([True]*len(each.x))
    #Update the following using solver_settings defined above
    for key, val in solver_settings.items():
        getattr(solver.optimizer,key)(val)
    #update mask info
    model.data = copy.deepcopy(model.data_original)
    [each.apply_mask() for each in model.data]
    if RAXS_FIT:
        raxs_tag, all_tag = get_data_type_tag(model)
        for i in range(len(raxs_tag)):
            model = set_model(model, int(raxs_tag[i]-100), all_tag.index(raxs_tag[i]))
            #simulate the model first
            print('Starting the RAXS fit, trial {} of {} trials in total'.format(i, len(raxs_tag)))
            print("Simulating the model now ...")
            model.simulate()
            print("Start the fit...")
            solver.StartFit()
            model.save(each_file)
    else:
        #simulate the model first
        print("Simulating the model now ...")
        model.simulate()
        print("Start the fit...")
        solver.StartFit()
        model.save(each_file)
        cov = solver.optimizer.return_covariance_matrix(fom_level = 0.2)
        model.save_addition('covariance_matrix',cov)
        sensitivity = solver.optimizer.return_sensitivity(max_epoch = 200, epoch_step = 0.1)
        model.save_addition('sensitivity',str(list(sensitivity)))
        model.save_addition_from_optimizer(each_file, solver.optimizer)
        # print(cov, sensitivity)
    
    
    
