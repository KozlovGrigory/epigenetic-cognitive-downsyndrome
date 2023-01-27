import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatterplot_cognitive_methylation (directory_folder, cognitive_indicator, methylation_frame, cognitive_frame, correlation_frame):

    cpg_name = list(correlation_frame['cpg'])[0]
    plt.figure(figsize = (6, 6))
    plt.plot(methylation_frame[cpg_name], cognitive_frame[cognitive_indicator], 'o', color = '#3366CC')

    m, b = np.polyfit(methylation_frame[cpg_name], cognitive_frame[cognitive_indicator], 1)
    plt.plot(methylation_frame[cpg_name], m * methylation_frame[cpg_name] + b, color = 'red')

    plt.title('Correlation coef.: {0:0.2f}, p-value: {1}, \nregression line slope: {2:0.2f}'.format(list(correlation_frame['correlation'])[0], list(correlation_frame['p-value'])[0], m), fontsize = 12)
    plt.xlabel(cpg_name, fontsize = 13)
    plt.ylabel(cognitive_indicator, fontsize = 13)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.savefig('{0}/{1}.png'.format(directory_folder, cognitive_indicator), dpi = 300)