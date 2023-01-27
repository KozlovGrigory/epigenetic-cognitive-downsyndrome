import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

def fdr_correction (directory_folder, cognitive_indicator, correlation_frame):

    correlation_frame_corr = fdrcorrection(correlation_frame['p-value'], alpha = 0.001, method = 'indep', is_sorted = True)
    correlation_frame_fdr = pd.DataFrame({'rejected': correlation_frame_corr[0], 
                                          'pvalue-corrected': correlation_frame_corr[1], 
                                          'pvalue-original': correlation_frame['p-value']})
    correlation_frame_fdr.to_csv('{0}/{1}_fdr.txt'.format(directory_folder, cognitive_indicator), sep = '\t', index = False)
    return correlation_frame_fdr