from pathlib import Path
import sys, os
import shutil

DaFy_path = Path(__file__).parent.parent.parent
Engine_pool_path = Path(__file__).parent.parent / 'EnginPool'
Filter_pool_path = Path(__file__).parent.parent / 'FilterPool'
util_path = Path(__file__).parent
xrv_path = DaFy_path / 'projects' / 'xrv'
dc_path = DaFy_path / 'projects' / 'orcalc'
ubmate_path = DaFy_path / 'projects' / 'ubmate'
ctr_path = DaFy_path / 'projects' / 'ctr'
superrod_path = DaFy_path / 'projects' / 'superrod'
viewer_path = DaFy_path / 'projects' / 'viewer'
superrod_batch_path = Path(__file__).parent.parent.parent / 'resources' / 'batchfile'
user_data_path = Path.home() / 'dafyAppData' / 'dump_files'
user_config_path = Path.home() / 'dafyAppData' / 'config_files'
user_example_path = Path.home() / 'dafyAppData' / 'examples'
superrod_path_list = map(lambda x: str(x), [Engine_pool_path, Filter_pool_path, util_path, superrod_path / 'core'])

user_data_path.mkdir(parents = True, exist_ok = True)
user_config_path.mkdir(parents = True, exist_ok = True)
user_example_path.mkdir(parents = True, exist_ok = True)

shutil.copytree(str(DaFy_path / 'resources' / 'example'), str(user_example_path), dirs_exist_ok=True)