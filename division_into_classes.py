import numpy as np
import pandas as pd

def division_boundary_values (lim1, lim2, lim3, cognitive_frame, cognitive_indicator):
    classes = pd.DataFrame({'value': list(cognitive_frame[cognitive_indicator]), 'class': 0})
    for i in range(len(classes['value'])):
        if classes['value'].loc[i] <= lim1:
            classes.loc[i, 'class'] = 0
        elif classes['value'].loc[i] > lim1 and classes['value'].loc[i] <= lim2:
            classes.loc[i, 'class'] = 1
        else:
            classes.loc[i, 'class'] = 2
    classes_equal = []  
    classes_equal.extend([list(classes['class']).count(0),
                        list(classes['class']).count(1),
                        list(classes['class']).count(2)])
    return classes, classes_equal

def division_into_classes (cognitive_indicator, cognitive_frame, user_choice, 
                           lim1, lim2, lim3):
    if user_choice == 0:
        boundary_values_all = []
        values = sorted(list(cognitive_frame[cognitive_indicator]))
        lim1 = values[len(values) // 3]
        lim2, lim3 = values[len(values) // 3 * 2], values[len(values) - 1]
        boundary_values_all.extend([lim1, lim2, lim3])
        classes_all, classes_all_equal = division_boundary_values(lim1, lim2, lim3, cognitive_frame, cognitive_indicator)
    
        boundary_values_unique = []
        values = sorted(list(set(cognitive_frame[cognitive_indicator])))
        lim1 = values[len(values) // 3]
        lim2, lim3 = values[len(values) // 3 * 2], values[len(values) - 1] 
        boundary_values_unique.extend([lim1, lim2, lim3])
        classes_unique, classes_unique_equal = division_boundary_values(lim1, lim2, lim3, cognitive_frame, cognitive_indicator)
    
        return boundary_values_all, boundary_values_unique, classes_all, classes_unique, classes_all_equal, classes_unique_equal
    elif user_choice == 1:
        boundary_values_all = []
        values = sorted(list(cognitive_frame[cognitive_indicator]))
        lim1 = values[len(values) // 3]
        lim2, lim3 = values[len(values) // 3 * 2], values[len(values) - 1]
        boundary_values_all.extend([lim1, lim2, lim3])
        classes_all, classes_all_equal = division_boundary_values(lim1, lim2, lim3, cognitive_frame, cognitive_indicator)
        return classes_all
    elif user_choice == 2:
        boundary_values_unique = []
        values = sorted(list(set(cognitive_frame[cognitive_indicator])))
        lim1 = values[len(values) // 3]
        lim2, lim3 = values[len(values) // 3 * 2], values[len(values) - 1] 
        boundary_values_unique.extend([lim1, lim2, lim3])
        classes_unique, classes_unique_equal = division_boundary_values(lim1, lim2, lim3, cognitive_frame, cognitive_indicator)
        return classes_unique  
    elif user_choice == 3:
        classes_user, classes_user_equal = division_boundary_values(lim1, lim2, lim3, cognitive_frame, cognitive_indicator)
        return classes_user 