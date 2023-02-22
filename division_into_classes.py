import numpy as np
import pandas as pd

def division_boundary_values (cognitive_indicator, cognitive_frame, class_values, lim1 = 0, lim2 = 0):
    if lim1 == 0 and lim2 == 0:
        splits = np.array_split(class_values, 3)
        lim1 = splits[0][-1]
        lim2 = splits[1][-1]
    
    classes = pd.DataFrame({'id': list(cognitive_frame.index), 'value': list(cognitive_frame[cognitive_indicator]), 'class': 0})  
    for i in range(len(classes['value'])):
        if classes['value'].loc[i] <= lim1:
            classes.loc[i, 'class'] = 0
        elif classes['value'].loc[i] > lim1 and classes['value'].loc[i] <= lim2:
            classes.loc[i, 'class'] = 1
        else:
            classes.loc[i, 'class'] = 2
    classes_members = [list(classes['class']).count(0),
                       list(classes['class']).count(1),
                       list(classes['class']).count(2)]
    classes.set_index('id', inplace = True)
    return classes, classes_members

def division_into_classes (cognitive_indicator, cognitive_frame, division_method = 0, lim1 = 0, lim2 = 0):
    if division_method != 3:
        class_values = sorted(list(cognitive_frame[cognitive_indicator]))
        classes_all, classes_all_members = division_boundary_values(cognitive_indicator, cognitive_frame, class_values)
        if division_method == 1:
            return classes_all
        class_values = sorted(list(set(cognitive_frame[cognitive_indicator])))
        classes_unique, classes_unique_members = division_boundary_values(cognitive_indicator, cognitive_frame, class_values)
        if division_method == 2:
            return classes_unique 
        return classes_all_members, classes_unique_members
    else:
        class_values = sorted(list(cognitive_frame[cognitive_indicator]))
        classes_user, classes_user_members = division_boundary_values(cognitive_indicator, cognitive_frame, class_values, lim1, lim2)
        return classes_user 