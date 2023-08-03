# -*- coding: utf-8 -*-
from pyqtgraph.parametertree import Parameter, ParameterTree
import configparser
from pathlib import Path
from os import listdir
import logging
root = Path(__file__).parent.parent/ "resources" / "templates"

logger = logging.getLogger('widget.parameter')
logger.propagate = True
f_handler = logging.FileHandler('./log_temp/parameter_tree_log.log', mode = 'w')
f_handler.setFormatter(logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s : %(lineno)d'))
f_handler.setLevel(logging.DEBUG)
logger.addHandler(f_handler)

class SolverParameters(ParameterTree):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_type = None
        self.config_file = None

    def _build_pars(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        pars = []
        for section in config.sections():
            par = {'name': section, 'type': 'group', 'children': []}
            for key, item in config.items(section):
                par['children'].append( {"name":key,"type":["str","text"][int(len(item)>20)],"value":str(item)})
            pars.append(par)
        return Parameter.create(name='params', type='group', children=pars)

    def init_pars(self,config_file):
        logger.info('Initialize the parameter tree from config file!')
        self.config_file = str(root/config_file)
        pars = self._build_pars()
        self.setParameters(pars, showTop=False)
        self.par = pars

    def save_parameter(self):
        config = configparser.ConfigParser()
        sections = self.par.names.keys()
        for section in sections:
            sub_sections = self.par.names[section].names.keys()
            items = {}
            for each in sub_sections:
                items[each] = str(self.par[(section,each)])
            config[section] = items
        with open(self.config_file,'w') as config_file:
            config.write(config_file)

    def update_parameter_in_solver(self,parent):
        pass

