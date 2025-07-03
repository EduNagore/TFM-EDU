import os
import pandas as pd
from    importlib               import import_module
import pickle
import h5py

path_errors_all = 'data_predict_all'             
results = pd.DataFrame(columns = [
    'config_num',
    'NY_config', 'NY_divide_ages', 'NY_ext_type', 'NY_PCA', 'NY_time', 'NY_error',
    'MV_config', 'MV_distance', 'MV_ext_type', 'MV_time', 'MV_error',
    'TOTAL_time', 'ADD_time', 'TOTAL_error'
])

all_models = False
show_models = [25, 28, 29, 30, 31, 32, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48]
all_vars = False
show_vars = ['config_num',
             'NY_config', 'NY_divide_ages', 'NY_ext_type', 'NY_PCA',
             'MV_config', 'MV_ext_type',
             'ADD_time', 'TOTAL_error']

i = 0
for file_name in os.listdir(path_errors_all):
    if file_name.endswith('.pkl') and file_name.startswith('errors_all_'):
        
        config_num = int(file_name[11:-4])
        
        if not all_models:
            if config_num not in show_models:
                continue
        
        #print(f'Loading config {config_num}')
        
        load_path = f'data_predict_all/errors_all_{config_num}.pkl'
        with open(load_path, 'rb') as file:
            loaded_results = pickle.load(file)
        
        results.loc[i, 'config_num']  = config_num
        results.loc[i, 'NY_config']   = loaded_results['config_ny']
        results.loc[i, 'MV_config']   = loaded_results['config_mv']
        results.loc[i, 'TOTAL_time']  = loaded_results['total_time']
        results.loc[i, 'TOTAL_error'] = loaded_results['total_error']
        
        config_file_all = import_module(f"configs.next_year.config_{loaded_results['config_ny']}")
        config_file = config_file_all.configuration
        
        if config_file.fast:
            print(f'Fast computing for config {config_num}: Discaded')
            continue
        
        results.loc[i, 'NY_divide_ages'] = str(config_file.ages_division if config_file.divide_ages else [])
        txt = config_file.ext_type
        if txt in ['MW_sm', 'OM_sm']:
            txt = txt + f'_{config_file.sigma}'
        results.loc[i, 'NY_ext_type'] = txt
        txt = config_file.PCA
        if txt in ['first_R']:
            txt = txt + f'_{config_file.PCA_proportion}'
        results.loc[i, 'NY_PCA'] = txt
        
        with h5py.File(f"data_pred_next_year/predictions_next_year_config_{loaded_results['config_ny']}_dict.h5", 'r') as file:
            results.loc[i, 'NY_time'] = file['time'][()]
            results.loc[i, 'NY_error'] = file['error'][()]
            
        del config_file, config_file_all
        
        config_file_all = import_module(f"configs.market_value.config_{loaded_results['config_mv']}")
        config_file = config_file_all.configuration
        
        if config_file.fast:
            print(f'Fast computing for config {config_num}: Discaded')
            continue
        
        results.loc[i, 'MV_distance'] = str(f'{config_file.norm} norm.{config_file.normalize}' if config_file.normalize else f'{config_file.norm}')
        results.loc[i, 'MV_ext_type'] = config_file.ext_type
        
        with h5py.File(f"data_pred_mvalue/info_mvalue_{loaded_results['config_mv']}_dict.h5", 'r') as file:
            results.loc[i, 'MV_time'] = file['time'][()]
            results.loc[i, 'MV_error'] = file['error'][()]
            
        results.loc[i, 'ADD_time']    = results.loc[i, 'NY_time'] + results.loc[i, 'TOTAL_time']
            
        del config_file, config_file_all
               
        i = i + 1
        
results.config_num = results.config_num.astype(int)
#results = results.sort_values(by='config_num')
results = results.sort_values(by=['NY_PCA', 'NY_ext_type', 'MV_ext_type'])
if all_vars:
    print(results)
else:
    print(results[show_vars])