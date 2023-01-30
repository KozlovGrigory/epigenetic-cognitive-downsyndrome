import pandas as pd
import numpy as np
import os

def data_preproc (files, directory_research, cognitive_indicator):
    indicator_folder = directory_research + '/' + cognitive_indicator
    if not os.path.isdir(indicator_folder):
        os.mkdir(indicator_folder)
        
    cpg_values = np.load(files['cpg_values'], 'r')['X'][0:29]
    cpg_names = np.load(files['cpg_names'], 'r')['all_cpg_names']
    data_samples = list(pd.read_csv(files['data_samples'], sep = '\t')['sample_name'][0:29])
    data_sex = list(pd.read_csv(files['data_samples'], sep = '\t')['gender'][0:29])
    cognitive_frame = pd.DataFrame(pd.read_csv(files['cognitive_frame'], sep = '\t', index_col = 0)[cognitive_indicator])
    
    methylation_frame = pd.DataFrame(cpg_values, index = data_samples, columns = cpg_names)
    cognitive_frame['sex'] = data_sex
    
    nan_index = cognitive_frame.index[pd.isnull(cognitive_frame[cognitive_indicator])]
    for nan in nan_index:
        cognitive_frame.drop([nan], inplace = True)
        methylation_frame.drop([nan], inplace = True)

    methylation_frame.to_csv('{0}/methylation_frame.txt'.format(indicator_folder), sep = '\t', index = True)
    cognitive_frame.to_csv('{0}/cognitive_frame.txt'.format(indicator_folder), sep = '\t', index = True)
    return methylation_frame, cognitive_frame, indicator_folder