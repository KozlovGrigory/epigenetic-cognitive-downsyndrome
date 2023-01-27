import pandas as pd
import numpy as np
import os

def data_preproc (cognitive_indicator):
    GSE52588_values = np.load('D:/unn/GSE52588/GSE52588_beta_fn.npz', 'r')['X'][0:29]
    GSE52588_names = np.load('D:/unn/GSE52588/GSE52588_beta_fn.npz', 'r')['all_cpg_names']
    GSE52588_samples = list(pd.read_csv('D:/unn/GSE52588/GSE52588_samples.txt', sep = '\t')['sample_name'][0:29])
    methylation_frame = pd.DataFrame(GSE52588_values, index = GSE52588_samples, columns = GSE52588_names)

    cognitive_frame = pd.DataFrame(pd.read_csv('D:/unn/GSE52588/DOWN_FENOTIPO_No4,8,12_PerCorrelazioni.tsv', 
                                           sep = '\t', index_col = 0)[cognitive_indicator])
    nan_index = cognitive_frame.index[pd.isnull(cognitive_frame[cognitive_indicator])]
    for i in nan_index:
        cognitive_frame.drop([i], inplace = True)
        methylation_frame.drop([i], inplace = True)

    directory_folder = 'D:/unn/test/{}'.format(cognitive_indicator)
    if not os.path.isdir(directory_folder):
        os.mkdir(directory_folder)
    methylation_frame.to_csv('D:/unn/test/{0}/{1}_methylation_frame.txt'.format(cognitive_indicator, cognitive_indicator), 
                             sep = '\t', index = True)
    cognitive_frame.to_csv('D:/unn/test/{0}/{1}_cognitive_frame.txt'.format(cognitive_indicator, cognitive_indicator), 
                           sep = '\t', index = True)
    return directory_folder, methylation_frame, cognitive_frame