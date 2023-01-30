import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

def scatterplot_cognitive_methylation (indicator_folder, cognitive_indicator, methylation_frame, cognitive_frame, correlation_frame):
    cpg_name = correlation_frame.index[0]
    color_labels = cognitive_frame['sex'].unique()
    rgb_values = [sns.color_palette('Set1')[1], sns.color_palette('Set1')[2]]
    color_map = dict(zip(color_labels, rgb_values))
    
    x, y = 15, 12
    fig_inch = (x/2.54, y/2.54)
    fig, ax = plt.subplots(figsize = fig_inch)
    ax.scatter(methylation_frame[cpg_name], cognitive_frame[cognitive_indicator], c = cognitive_frame['sex'].map(color_map))
    handles = [Line2D([0], [0], marker = 'o', color = 'w', 
                      markerfacecolor = v, label = k, markersize = 10) for k, v in color_map.items()]
    ax.legend(title = 'Sex:', handles = handles, bbox_to_anchor = (1.05, 1), 
              loc = 'upper left', fontsize = 12, title_fontsize = 12)

    m, b = np.polyfit(methylation_frame[cpg_name], cognitive_frame[cognitive_indicator], 1)
    plt.plot(methylation_frame[cpg_name], m * methylation_frame[cpg_name] + b, color = sns.color_palette('Set1')[0])

    ax.set_title('Correlation coef.: {0:0.2f}, p-value: {1:e}, \nregression line slope: {2:0.2f}'.format(list(correlation_frame['correlation'])[0], 
                                                                                                         list(correlation_frame['p-value-adjusted'])[0], 
                                                                                                         m), fontsize = 12)
    ax.set_xlabel(cpg_name, fontsize = 12)
    ax.set_ylabel(cognitive_indicator, fontsize = 12)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.tight_layout()
    plt.savefig('{0}/{1}_png.png'.format(indicator_folder, cognitive_indicator), dpi = 300)
    plt.savefig('{0}/{1}_svg.svg'.format(indicator_folder, cognitive_indicator), dpi = 300)
    plt.savefig('{0}/{1}_tif.tif'.format(indicator_folder, cognitive_indicator), dpi = 300)
    plt.savefig('{0}/{1}_svg.svg'.format(indicator_folder, cognitive_indicator), dpi = 300)
    plt.close()
    
def genes_cpgs (indicator_folder, files, cognitive_indicator, correlation_frame):
    cpgs_annotations = pd.read_csv(files['cpgs_annotations'], sep = '\t', index_col = 0)
    cpgs_list1 = []
    cpgs_list2 = []
    cpgs_list3 = []
    for a in correlation_frame.index:
        cpg = cpgs_annotations['UCSC_REFGENE_NAME'][a]
        cpgs_list1.append(cpg)
    cpgs_list2 = [x for x in cpgs_list1 if x == x]
    for b in cpgs_list2:
        cpgs = b.split(';')
        for c in cpgs:
            if c not in cpgs_list3:
                cpgs_list3.append(c)
    cpgs_list_result = open('{0}/genes.txt'.format(indicator_folder), 'w')
    for i in cpgs_list3:
        cpgs_list_result.write(i + '\n')
    cpgs_list_result.close()
    
def correlation_cognitive_methylation (alpha, files, indicator_folder, cognitive_indicator, methylation_frame, cognitive_frame):
    correlation_coef = []
    pvalues = []
    for cpg_name in tqdm(methylation_frame.columns, ncols = 100):
        corr = spearmanr(cognitive_frame[cognitive_indicator], methylation_frame[cpg_name])[0]
        correlation_coef.append(corr)  
        pval = spearmanr(cognitive_frame[cognitive_indicator], methylation_frame[cpg_name])[1]
        pvalues.append(pval)
    correlation_frame = pd.DataFrame({'cpg': methylation_frame.columns, 'correlation': correlation_coef, 'p-value': pvalues})
    correlation_frame = correlation_frame.set_index('cpg') 
    correlation_frame = correlation_frame.loc[(correlation_frame['p-value'] < alpha)]
    correlation_frame = correlation_frame.sort_values(by = 'p-value', ascending = True)
    
    pvalue_adjusted = fdrcorrection(correlation_frame['p-value'], alpha = alpha, method = 'indep', is_sorted = True)
    correlation_frame['p-value-adjusted'] = pvalue_adjusted[1]
    correlation_frame = correlation_frame.loc[(correlation_frame['p-value-adjusted'] < alpha)]
    scatterplot_cognitive_methylation (indicator_folder, cognitive_indicator, 
                                       methylation_frame, cognitive_frame, correlation_frame)
    genes_cpgs (indicator_folder, files, cognitive_indicator, correlation_frame)
    
    correlation_frame.to_csv('{0}/correlation_frame.txt'.format(indicator_folder), sep = '\t', index = True)
    return correlation_frame