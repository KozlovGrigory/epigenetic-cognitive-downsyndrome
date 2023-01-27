import pandas as pd

def genes_cpgs (directory_folder, cognitive_indicator, correlation_frame):

    cpgs_annotations = pd.read_csv('D:/unn/GSE52588/cpgs_annotations.txt', sep = '\t', index_col = 0)
    cpgs_correlation = list(correlation_frame['cpg'])

    cpgs_list1 = []
    cpgs_list2 = []
    cpgs_list3 = []

    for a in cpgs_correlation:
        cpg = cpgs_annotations['UCSC_REFGENE_NAME'][a]
        cpgs_list1.append(cpg)
    cpgs_list2 = [x for x in cpgs_list1 if x == x]

    for b in cpgs_list2:
        cpgs = b.split(';')
        for c in cpgs:
            if c not in cpgs_list3:
                cpgs_list3.append(c)

    cpgs_list_result = open('{0}/{1}_genes.txt'.format(directory_folder, cognitive_indicator), 'w')
    for i in cpgs_list3:
        cpgs_list_result.write(i + '\n')
    cpgs_list_result.close()
    return len(cpgs_list3)