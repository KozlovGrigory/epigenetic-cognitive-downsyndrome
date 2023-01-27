import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr

def correlation_cognitive_methylation (alpha, directory_folder, cognitive_indicator, methylation_frame, cognitive_frame):
    correlation_coef = []
    pvalues = []
    for cpg_name in methylation_frame.columns:
        corr = spearmanr(cognitive_frame[cognitive_indicator], methylation_frame[cpg_name])[0]
        correlation_coef.append(corr)  
        pval = spearmanr(cognitive_frame[cognitive_indicator], methylation_frame[cpg_name])[1]
        pvalues.append(pval)
    correlation_frame = pd.DataFrame({'cpg': methylation_frame.columns})
    correlation_frame.loc[:, 'p-value'] = pvalues
    correlation_frame.loc[:, 'correlation'] = correlation_coef
    #correlation_frame = correlation_frame.set_index('cpg')
    
    correlation_frame = correlation_frame.loc[(correlation_frame['p-value'] < alpha)]
    correlation_frame = correlation_frame.sort_values(by = 'p-value', ascending = True)
    correlation_frame.to_csv('{0}/{1}_correlation_frame.txt'.format(directory_folder, cognitive_indicator), sep = '\t', index = True)
    return correlation_frame