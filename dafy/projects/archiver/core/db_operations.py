from .util import error_pop_up, PandasModel
from ..config import config
from ..widgets.dialogues import NewProject,LoginDialog, RegistrationDialog
from dotenv import load_dotenv
from pathlib import Path
import os, certifi, datetime, time
from pymongo import MongoClient
from PyQt5.QtWidgets import QMessageBox
from pathlib import Path
import PyQt5
import pandas as pd
from functools import partial
import logging

logger = logging.getLogger(__name__)
logger.propagate = True
f_handler = logging.FileHandler('db_operation.log', mode = 'w')
f_handler.setFormatter(logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s : %(lineno)d'))
f_handler.setLevel(logging.DEBUG)
logger.addHandler(f_handler)

def start_mongo_client_cloud(self):
    try:
        if not os.path.exists(str(Path(__file__).parent.parent/ "resources" / "private" / "atlas_password.dot")):
            error_pop_up('You should create a file named atlas_password under Library_Manager/resources/private folder, \
                            where you save the atlas url link for your MongoDB atlas cloud account. \
                            please use the format ATLAS_URL="URL LINK"')
        else:
            env = load_dotenv(str(Path(__file__).parent.parent/ "resources" / "private" / "atlas_password.dot"))
            if env:
                url = os.getenv('ATLAS_URL') 
                if not url:
                    logger.error('could not load enviroment variable from atlas_password.dot')
                    print('something is wrong')
                else:
                    logger.info('fire up login dialog')
                    login_dialog(self, url)
                    logger.info('finish login operation.')
                    #self.mongo_client = MongoClient(url,tlsCAFile=certifi.where())
    except Exception as e:
        error_pop_up('Fail to start mongo client.'+'\n{}'.format(str(e)),'Error')

def register_new_user(self):
    try:
        if not os.path.exists(str(Path(__file__).parent.parent/ "resources" / "private" / "atlas_password.dot")):
            error_pop_up('You should create a file named atlas_password under Library_Manager/resources/private folder, \
                            where you save the atlas url link for your MongoDB atlas cloud account. \
                            please use the format ATLAS_URL="URL LINK"')
        else:
            env = load_dotenv(str(Path(__file__).parent.parent/ "resources" / "private" / "atlas_password.dot"))
            if env:
                url = os.getenv('ATLAS_URL') 
                register_dialog(self, url)
                #self.mongo_client = MongoClient(url,tlsCAFile=certifi.where())
            else:
                url = ''
                print('something is wrong')
    except Exception as e:
        error_pop_up('Fail to start mongo client for new user registration.'+'\n{}'.format(str(e)),'Error')

def extract_project_info(self):
    all = self.database.project_info.find()[0]
    self.plainTextEdit_project_info.setPlainText(all['project_info'])
    try:
        self.lineEdit_energy.setText(all['energy'])
        self.lineEdit_beamsize.setText(all['beamsize'])
        self.lineEdit_beamflux.setText(all['beamflux'])
        self.lineEdit_beam_center.setText(all['beamCenter'])
        self.lineEdit_mode.setText(all['mode'])
        self.lineEdit_geometry.setText(all['geometry'])
        self.lineEdit_incidence_angle.setText(all['incidenceAngle'])
        self.lineEdit_extraInfo.setText(all['extraInfo'])
    except:
        pass

def load_project(self):
    self.database = self.mongo_client[self.comboBox_project_list.currentText()]
    extract_project_info(self)
    #self.update_paper_list_in_listwidget()
    init_pandas_model_from_db(self)
    update_sample_list_in_combobox(self)
    extract_sample_info(self)

def update_project_info(self):
    try:
        old = self.database.project_info.find()[0]
        self.database.project_info.replace_one(old,{'project_info':self.plainTextEdit_project_info.toPlainText(),
                                            'energy':self.lineEdit_energy.text(),
                                            'beamsize':self.lineEdit_beamsize.text(),
                                            'beamflux':self.lineEdit_beamflux.text(),
                                            'beamCenter':self.lineEdit_beam_center.text(),
                                            'mode':self.lineEdit_mode.text(),
                                            'geometry':self.lineEdit_geometry.text(),
                                            'incidenceAngle':self.lineEdit_incidence_angle.text(),
                                            'extraInfo':self.lineEdit_extraInfo.text(),})
        error_pop_up('Project information has been updated successfully!','Information')
    except Exception as e:
        error_pop_up('Failure to update Project information!','Error')

def new_project_dialog(self):
    dlg = NewProject(self)
    dlg.exec()

def login_dialog(self, url):
    dlg = LoginDialog(self, url)
    dlg.exec()

def logout(self):
    self.name = 'undefined'
    self.user_name = 'undefined'
    self.role = 'undefined'
    self.mongo_client = None
    self.database = None
    self.removeToolBar(self.toolBar)
    try:
        self.init_gui(self.ui)
        self.statusLabel.setText('Goodbye, you are logged out!')
    except:
        pass

def register_dialog(self, url):
    dlg = RegistrationDialog(self, url)
    dlg.exec()

def update_sample_list_in_combobox(self):
    samples = get_samples_in_a_list(self)
    self.comboBox_sample_ids.clear()
    self.comboBox_sample_ids.addItems(samples)

def get_samples_in_a_list(self):
    all_info = list(self.database.sample_info.find({},{'_id':0}))
    sample_id_list = [each['sample_id'] for each in all_info]
    return sorted(sample_id_list)

def get_scans_in_a_list(self):
    all_info = list(self.database.scan_info.find({'sample_id':self.comboBox_sample_ids.currentText()}))
    scan_list = [each['scan_id'] for each in all_info]
    return scan_list

def init_pandas_model_from_db(self, collection_records = None):
    if self.comboBox_sample_ids.currentText()=='':
        return
    data = {}
    for each in config.display_fields:
        data[each] = []
    #data = {'select':[],'scan_id':[],'scan_type':[],'otherInfo':[],'quality':[],'status':[],'note':[]}
    if collection_records==None:
        collection_records = self.database.scan_info.find({'sample_id':self.comboBox_sample_ids.currentText()})
    for each in collection_records:
        for key in data.keys():
            if key=='select':
                data[key].append(False)
            else:
                data[key].append(each[key])
    data = pd.DataFrame(data)
    if 'select' in config.display_fields:
        data['select'] = data['select'].astype(bool)
    self.pandas_model_scan_info = PandasModel(data = data, tableviewer = self.tableView_data_info, main_gui = self, rgb_bkg=(25,35,45),rgb_fg=(200,200,200))
    self.tableView_data_info.setModel(self.pandas_model_scan_info)
    self.tableView_data_info.resizeColumnsToContents()
    self.tableView_data_info.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
    self.tableView_data_info.horizontalHeader().setStretchLastSection(True)
    self.tableView_data_info.clicked.connect(partial(update_selected_scan_info,self))

def update_selected_scan_info(self, index = None):
    self.current_scan_id  = self.pandas_model_scan_info._data['scan_id'].tolist()[index.row()]
    sample_id = self.pandas_model_scan_info._data['sample_id'].tolist()[index.row()]
    if self.comboBox_sample_ids.currentText()!=sample_id:
        self.comboBox_sample_ids.setCurrentText(sample_id)
        extract_sample_info(self)
    extract_scan_info(self)

def update_selected_sample_info(self, index = None):
    self.comboBox_sample_ids.setCurrentText(self.pandas_model_sample_info._data['sample_id'].tolist()[index.row()])
    extract_sample_info(self)

def update_project_info(self):
    try:
        self.database.project_info.drop()
        beamtime_info = {'project_info':self.plainTextEdit_project_info.toPlainText(),
                         'energy':self.lineEdit_energy.text(),
                         'beamsize':self.lineEdit_beamsize.text(),
                         'beamflux':self.lineEdit_beamflux.text(),
                         'beamCenter':self.lineEdit_beam_center.text(),
                         'mode':self.lineEdit_mode.text(),
                         'geometry':self.lineEdit_geometry.text(),
                         'incidenceAngle':self.lineEdit_incidence_angle.text(),
                         'extraInfo':self.lineEdit_extraInfo.text(),}
        self.database.project_info.insert_one(beamtime_info)
        error_pop_up('Project information has been updated successfully!','Information')
    except Exception as e:
        error_pop_up('Failure to update Project information!','Error')

def clear_all_fields(self):
    self.lineEdit_scan_id.setText(''),
    self.lineEdit_scan_number.setText(''),
    self.lineEdit_scan_points.setText(''),
    self.lineEdit_scan_file.setText(''),
    self.lineEdit_scan_type.setText(''),
    self.lineEdit_scan_macro.setText(''),
    self.lineEdit_other_info.setText(''),
    self.lineEdit_status.setText(''),
    self.lineEdit_experimenter.setText(''),
    self.lineEdit_analysis_type.setText(''),
    self.lineEdit_time.setText(''),
    self.lineEdit_location.setText(''),
    self.lineEdit_analysis_tool.setText(''),
    self.lineEdit_data_quality.setText(''),
    self.lineEdit_performer.setText(''),
    self.lineEdit_extraInfo_analysis.setText(''),
    self.textEdit_comments.setPlainText(''), 

def extract_sample_info(self):
    sample_id = self.comboBox_sample_ids.currentText()
    if sample_id=='':
        #clean the fields first
        self.lineEdit_sample_ID.setText('')
        self.lineEdit_preparation.setText('')
        self.lineEdit_rxn.setText('')
        self.lineEdit_formular.setText('')
        self.lineEdit_other_sample_info.setText('')
        self.textEdit_des.setPlainText('')     
        return
    target = self.database.sample_info.find_one({'sample_id':sample_id})
    self.current_scan_id = None
    scans = self.database.scan_info.find({'sample_id':sample_id})
    try:
        self.current_scan_id = scans[0]['scan_id']
    except:#no scan info records yet
        clear_all_fields(self)
        return
    sample_info = {'sample_id':self.lineEdit_sample_ID.setText,
                    'sample_preparation':self.lineEdit_preparation.setText,
                    'rxn_condition':self.lineEdit_rxn.setText,
                    'chemical_formular':self.lineEdit_formular.setText,
                    'other_info':self.lineEdit_other_sample_info.setText,
                    'general_des':self.textEdit_des.setPlainText,
                    }    
    for key, item in sample_info.items():
        if key in target:
            item(target[key])
        else:
            if key == 'graphical_abstract':
                pass
            else:
                item('')

def extract_scan_info(self):
    scan_id = self.current_scan_id
    if scan_id == None:
        return
    target = self.database.scan_info.find_one({'scan_id':scan_id})
    scan_info = {   'scan_id': self.lineEdit_scan_id.setText,
                    'scan_number':self.lineEdit_scan_number.setText,
                    'scan_points':self.lineEdit_scan_points.setText,
                    'scan_file':self.lineEdit_scan_file.setText,
                    'scan_type':self.lineEdit_scan_type.setText,
                    'scan_macro':self.lineEdit_scan_macro.setText,
                    'other_info':self.lineEdit_other_info.setText,
                    'status':self.lineEdit_status.setText,
                    'time':self.lineEdit_time.setText,
                    'experimenter':self.lineEdit_experimenter.setText,
                    'note':self.lineEdit_note.setText,
                    'analysis_type':self.lineEdit_analysis_type.setText,
                    'data_location':self.lineEdit_location.setText,
                    'analysis_tool':self.lineEdit_analysis_tool.setText,
                    'data_quality':self.lineEdit_data_quality.setText,
                    'analysis_performer':self.lineEdit_performer.setText,
                    'analysis_extraInfo':self.lineEdit_extraInfo_analysis.setText,
                    'comment':self.textEdit_comments.setPlainText,                    
                    }    
    for key, item in scan_info.items():
        if key in target:
            item(target[key])
        else:
            item('')

def delete_one_scan(self):
    scan_id = self.current_scan_id
    reply = QMessageBox.question(self, 'Message', 'Are you sure to delete this scan record?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if reply == QMessageBox.Yes:
        try:
            self.database['scan_info'].delete_many({'scan_id':scan_id})
            self.statusbar.clearMessage()
            self.statusbar.showMessage('The scan record is deleted from DB successfully:-)')
            # self.update_paper_list_in_listwidget()
            #update_sample_list_in_combobox(self)
            init_pandas_model_from_db(self)
            extract_scan_info(self)
        except:
            error_pop_up('Fail to delete the paper info!','Error')

def delete_one_sample(self):
    sample_id = self.comboBox_sample_ids.currentText()
    reply = QMessageBox.question(self, 'Message', 'Are you sure to delete this sample record?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if reply == QMessageBox.Yes:
        try:
            self.database['scan_info'].delete_many({'sample_id':sample_id})
            self.database['sample_info'].delete_many({'sample_id':sample_id})
            self.statusbar.clearMessage()
            self.statusbar.showMessage('The scan record is deleted from DB successfully:-)')
            # self.update_paper_list_in_listwidget()
            update_sample_list_in_combobox(self)
            extract_sample_info(self)
            init_pandas_model_from_db(self)
        except:
            error_pop_up('Fail to delete the paper info!','Error')

def update_scan_info(self, message='Would you like to update your database with new input?', status_msg = 'Update the scan info successfully:-)'):
    scan_id = self.lineEdit_scan_id.text()
    original = self.database.scan_info.find_one({'scan_id':scan_id})
    new_scan_info = {'sample_id':self.comboBox_sample_ids.currentText(),
                    'scan_id':self.lineEdit_scan_id.text(),
                    'scan_number':self.lineEdit_scan_number.text(),
                    'scan_points':self.lineEdit_scan_points.text(),
                    'scan_file':self.lineEdit_scan_file.text(),
                    'scan_type':self.lineEdit_scan_type.text(),
                    'scan_macro':self.lineEdit_scan_macro.text(),
                    'other_info':self.lineEdit_other_info.text(),
                    'status':self.lineEdit_status.text(),
                    'time':datetime.datetime.today().strftime('%Y-%m-%d'),
                    'experimenter':self.lineEdit_experimenter.text(),
                    'note':self.lineEdit_note.text(),
                    'analysis_type':self.lineEdit_analysis_type.text(),
                    'data_location':self.lineEdit_location.text(),
                    'analysis_tool':self.lineEdit_analysis_tool.text(),
                    'data_quality':self.lineEdit_data_quality.text(),
                    'analysis_performer':self.lineEdit_performer.text(),
                    'analysis_extraInfo':self.lineEdit_extraInfo_analysis.text(),
                    'comment':self.textEdit_comments.toPlainText(),
                    }    
    try:        
        reply = QMessageBox.question(self, 'Message', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.database.scan_info.replace_one(original,new_scan_info)
            #self.update_paper_list_in_listwidget()
            init_pandas_model_from_db(self)
            self.statusbar.clearMessage()
            self.statusbar.showMessage(status_msg)
            return True
        else:
            return False
    except Exception as e:
        error_pop_up('Fail to update record :-(\n{}'.format(str(e)),'Error')     
    return False

def update_sample_info(self, parser = None):
    sample_id = self.lineEdit_sample_ID.text()
    original = self.database.sample_info.find_one({'sample_id':sample_id})    
    sample_info = {'sample_id':self.lineEdit_sample_ID.text(),
                    'sample_preparation':self.lineEdit_preparation.text(),
                    'rxn_condition':self.lineEdit_rxn.text(),
                    'chemical_formular':self.lineEdit_formular.text(),
                    'extra_info':self.lineEdit_other_sample_info.text(),
                    'general_des':self.textEdit_des.toPlainText(),
                    }
    if parser!=None:
        sample_info = parser

    # if os.path.exists(self.lineEdit_pdf.text()):
        # self.file_worker.insertFile(filePath = self.lineEdit_pdf.text(),paper_id = paper_id_temp)
    try:
        self.database.sample_info.replace_one(original, sample_info)
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Update the sample info sucessfully!')
    except Exception as e:
        error_pop_up('Failure to append sample info! Due to:\n{}'.format(str(e)),'Error')    

#create a new paper record in database
def add_sample_info(self, parser = None):
    sample_info = {'sample_id':self.lineEdit_sample_ID.text(),
                    'sample_preparation':self.lineEdit_preparation.text(),
                    'rxn_condition':self.lineEdit_rxn.text(),
                    'chemical_formular':self.lineEdit_formular.text(),
                    'extra_info':self.lineEdit_other_sample_info.text(),
                    'general_des':self.textEdit_des.toPlainText(),
                    }
    if parser!=None:
        sample_info = parser
    samples = get_samples_in_a_list(self)
    if sample_info['sample_id'] in samples:
        error_pop_up('The sample id is already used. Please choose a different one.')
        return
    #paper_info['archive_date'] = datetime.datetime.today().strftime('%Y-%m-%d')

    # if os.path.exists(self.lineEdit_pdf.text()):
        # self.file_worker.insertFile(filePath = self.lineEdit_pdf.text(),paper_id = paper_id_temp)
    try:
        self.database.sample_info.insert_one(sample_info)
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Append the paper info sucessfully!')
        #self.update_paper_list_in_listwidget()
        update_sample_list_in_combobox(self)
        self.comboBox_sample_ids.setCurrentText(sample_info['sample_id'])
        init_pandas_model_from_db(self)
    except Exception as e:
        error_pop_up('Failure to append sample info! Due to:\n{}'.format(str(e)),'Error')              

def add_scan_info(self, parser = None):
    if self.comboBox_sample_ids.currentText()=='':
        error_pop_up('You should create a new sample first!')
        return
    scan_info = {'sample_id':self.comboBox_sample_ids.currentText(),
                    'scan_id':self.lineEdit_scan_id.text(),
                    'scan_number':self.lineEdit_scan_number.text(),
                    'scan_points':self.lineEdit_scan_points.text(),
                    'scan_file':self.lineEdit_scan_file.text(),
                    'scan_type':self.lineEdit_scan_type.text(),
                    'scan_macro':self.lineEdit_scan_macro.text(),
                    'other_info':self.lineEdit_other_info.text(),
                    'status':self.lineEdit_status.text(),
                    'time':datetime.datetime.today().strftime('%Y-%m-%d'),
                    'experimenter':self.lineEdit_experimenter.text(),
                    'note':self.lineEdit_note.text(),
                    'analysis_type':self.lineEdit_analysis_type.text(),
                    'data_location':self.lineEdit_location.text(),
                    'analysis_tool':self.lineEdit_analysis_tool.text(),
                    'data_quality':self.lineEdit_data_quality.text(),
                    'analysis_performer':self.lineEdit_performer.text(),
                    'analysis_extraInfo':self.lineEdit_extraInfo_analysis.text(),
                    'comment':self.textEdit_comments.toPlainText(),
                    }
    if parser!=None:
        scan_info = parser
    scans = get_scans_in_a_list(self)
    if scan_info['scan_id'] in scans:
        error_pop_up('The scan id is already used. Please choose a different one.')
        return
    #paper_info['archive_date'] = datetime.datetime.today().strftime('%Y-%m-%d')

    # if os.path.exists(self.lineEdit_pdf.text()):
        # self.file_worker.insertFile(filePath = self.lineEdit_pdf.text(),paper_id = paper_id_temp)
    try:
        self.database.scan_info.insert_one(scan_info)
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Append the scan info sucessfully!')
        #self.update_paper_list_in_listwidget()
        #update_scan_list_in_combobox(self)
        #self.comboBox_sample_ids.setCurrentText(sample_info['sample_id'])
        init_pandas_model_from_db(self)
    except Exception as e:
        error_pop_up('Failure to append sample info! Due to:\n{}'.format(str(e)),'Error')  

def load_processed_data_from_cloud(self):
    results = list(self.database.data_info.find())
    if len(results)==0:
        return
    results_pd = []
    for each in results:
        del each['_id']
        num_items = len(each[[each_key for each_key in list(each.keys()) if each_key!='scan_id'][0]])
        each['scan_id'] = [each['scan_id']]*num_items
        if 'select' not in each:
            each['select'] = [False]*num_items
        columns = list(each.keys())
        columns.remove('select')
        columns = ['select'] + columns
        results_pd.append(pd.DataFrame(each)[columns])
    if not hasattr(self, 'pandas_model_processed_data_info'):
        self.pandas_model_processed_data_info = PandasModel(data = pd.concat(results_pd, ignore_index=True), tableviewer = self.tableView_processed_data, main_gui = self, rgb_bkg=(25,35,45),rgb_fg=(200,200,200))
        self.tableView_processed_data.setModel(self.pandas_model_processed_data_info)
        self.tableView_processed_data.resizeColumnsToContents()
        self.tableView_processed_data.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        self.tableView_processed_data.horizontalHeader().setStretchLastSection(True)
    else:
        self.pandas_model_processed_data_info._data = pd.concat(results_pd, ignore_index=True)

def save_processed_data_to_cloud(self):
    #overwrite or not
    reply = QMessageBox.question(self, 'Message', 'Update processed data info to cloud?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if reply == QMessageBox.Yes:
        try:
            overwrite = self.radioButton_overwrite.isChecked()
            save_channels = self.lineEdit_channels.text().replace(' ','').rsplit(',')
            if save_channels==['']:
                save_channels = None
            save_channels, field_values, return_list = self.pandas_model_processed_data_info.get_dict_from_data(
                                                                        ref_key = 'scan_id', 
                                                                        selected_columns = save_channels)
            for val in field_values:
                scan_info = self.database.data_info.find_one({'scan_id':val})
                new_scan_info = return_list[field_values.index(val)]
                if scan_info != None:
                    if overwrite:
                        self.database.data_info.replace_one(scan_info, new_scan_info)
                    else:
                        temp_values = [{ '$each': new_scan_info[channel] } for channel in save_channels]
                        final_mongo_query = dict(zip(save_channels, temp_values))
                        self.database.data_info.update_one(
                            { 'scan_id': val },
                            { '$push': final_mongo_query })
                else:
                    self.database.data_info.insert_one(new_scan_info)
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Success to update processed data to cloud!')
        except Exception as e:
            error_pop_up('Failed to update to cloud due to:'+str(e))
    else:
        pass

def general_query_by_field(self, field, query_string, target_field, collection_name, database = None):
    """
    Args:
        field ([string]): in ['author','book_name','book_id','status','class']
        query_string ([string]): [the query string you want to perform, e.g. 1999 for field = 'year']
        target_filed([string]): the targeted filed you would like to extract
        collection_name([string]): the collection name you would like to target

    Returns:
        [list]: [value list of target_field with specified collection_name]
    e.g.
    general_query_by_field(self, field='name', query_string='jackey', target_field='email', collection_name='user_info')
    means I would like to get a list of email for jackey in user_info collection in the current database
    """

    if database == None:
        database = self.database
    index_name = database[collection_name].create_index([(field,'text')])
    targets = database[collection_name].find({"$text": {"$search": "\"{}\"".format(query_string)}})
    #drop the index afterwards
    return_list = [each[target_field] for each in targets]
    # self.database.paper_info.drop_index(index_name)
    database[collection_name].drop_index(index_name)
    return return_list  

def logical_query(self, collection, logical_opt, field_value_lf, field_value_rt, return_fields = None):
    if return_fields==None:
        return self.database[collection].find({
            f"${logical_opt}":[
                field_value_lf,
                field_value_rt
            ]
        })
    else:
        return self.database[collection].find({
            f"${logical_opt}":[
                field_value_lf,
                field_value_rt
            ]
        }, dict(zip(return_fields,[1]*len(return_fields))))

