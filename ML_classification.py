import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
import catboost as cb
import SHAP_interpretation as si
from sklearn import tree, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def confusion_matrices (X, y, best_models, models_names, cognitive_indicator, indicator_folder):            
    for model_opt in best_models:
        cv = LeaveOneOut()
        pred_targets = np.array([])
        act_targets = np.array([])    
        for train_ix, test_ix in cv.split(X):
            Xm_train, Xm_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            ym_train, ym_test = y.iloc[train_ix], y.iloc[test_ix]
            model_opt.fit(Xm_train, ym_train)    
            pred_labels = model_opt.predict(Xm_test)
            pred_targets = np.append(pred_targets, pred_labels)
            act_targets = np.append(act_targets, ym_test)

        acc_m = accuracy_score(act_targets, pred_targets)
        f1_m = f1_score(act_targets, pred_targets, average = 'macro')
        
        x, y = 15, 12
        fig_inch = (x/2.54, y/2.54)
        fig, ax = plt.subplots(figsize = fig_inch)
        cf_matrix = confusion_matrix(act_targets, pred_targets)
        ax = sns.heatmap(cf_matrix, annot = True, cmap = 'coolwarm')     
        ax.set_title('{0}\nAccuracy = {1:0.3f}, F1 = {2:0.3f}'.format(models_names[best_models.index(model_opt)], acc_m, f1_m), fontdict = {'size':'14'})
        ax.set_xlabel('Predicted', fontdict = {'size':'14'})
        ax.set_ylabel('Actual', fontdict = {'size':'14'})
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.xaxis.set_ticklabels(['Small','Medium', 'Large'])
        ax.yaxis.set_ticklabels(['Small','Medium', 'Large'])
        plt.tight_layout()
        plt.savefig('{0}/{1}_{2}_svg.svg'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), dpi = 300)
        plt.savefig('{0}/{1}_{2}_png.png'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), dpi = 300)
        plt.savefig('{0}/{1}_{2}_tif.tif'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), dpi = 300)
        plt.close()

def quality_metrics (y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = 'macro')
    return acc, f1

def Grid_Search_CV (model, params, X_train, X_test, 
                    y_train, y_test, acc_lst, f1_lst, 
                    best_params_lst, cv_err_lst):
    grid_cv = GridSearchCV(estimator = model, param_grid = params, cv = 5, n_jobs = -1)
    grid_cv.fit(X_train, y_train)
    y_pred = grid_cv.predict(X_test)
    best_params, cv_err = grid_cv.best_params_, grid_cv.best_score_
    
    acc, f1 = quality_metrics (y_test, y_pred)
    acc_lst.append(acc)
    f1_lst.append(f1)
    best_params_lst.append(best_params)
    cv_err_lst.append(cv_err)
    
    return grid_cv, best_params_lst, cv_err_lst, acc_lst, f1_lst

def ML_classification (correlation_frame, methylation_frame, indicator_classes, cognitive_indicator, indicator_folder):
    indicator_folder = '{0}/ML_classification'.format(indicator_folder)
    if not os.path.isdir(indicator_folder):
            os.mkdir(indicator_folder)
            
    X = methylation_frame[correlation_frame.index]
    y = indicator_classes['class']
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 111)
    best_params_lst, cv_err_lst = [], []
    acc_lst, f1_lst = [], []
    
    dtree_model = tree.DecisionTreeClassifier(random_state = 111)
    params = {'max_depth': range (1, 4, 1)}
    dtree_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(dtree_model, params, X_train, X_test, 
                                                                                    y_train, y_test, acc_lst, f1_lst, 
                                                                                    best_params_lst, cv_err_lst)
    rf_model = RandomForestClassifier(random_state = 111)
    params = {'n_estimators': range(10, 110, 10)}
    rf_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(rf_model, params, X_train, X_test, 
                                                                                 y_train, y_test, acc_lst, f1_lst, 
                                                                                 best_params_lst, cv_err_lst)
    xg_model = xgb.XGBClassifier(objective ='multi:softprob', random_state = 111)
    params = {'max_depth': range (1, 4, 1), 'n_estimators': range(10, 110, 10), 
              'learning_rate': [0.1, 0.01, 0.05]}
    xgb.set_config(verbosity = 0)
    xg_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(xg_model, params, X_train, X_test, 
                                                                                 y_train, y_test, acc_lst, f1_lst, 
                                                                                 best_params_lst, cv_err_lst)
    cb_model = cb.CatBoostClassifier(loss_function = 'MultiClass', random_state = 111)
    params = {'iterations': [100, 150, 200], 'learning_rate': [0.03, 0.1], 
              'depth': [2, 3, 4], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    cb_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(cb_model, params, X_train, X_test, 
                                                                                 y_train, y_test, acc_lst, f1_lst, 
                                                                                 best_params_lst, cv_err_lst)
    
    lda_model = LDA()
    params = [{'solver': ['svd', 'lsqr', 'eigen']}]
    lda_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(lda_model, params, X_train, X_test, 
                                                                                  y_train, y_test, acc_lst, f1_lst, 
                                                                                  best_params_lst, cv_err_lst)
    qda_model = QDA()
    params = [{'reg_param': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}]
    qda_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(qda_model, params, X_train, X_test, 
                                                                                  y_train, y_test, acc_lst, f1_lst, 
                                                                                  best_params_lst, cv_err_lst)
    logist = LogisticRegression(solver = 'liblinear')
    params = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}]
    logist_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(logist, params, X_train, X_test, 
                                                                                     y_train, y_test, acc_lst, f1_lst, 
                                                                                     best_params_lst, cv_err_lst)
    svc_linear = svm.SVC(kernel = 'linear')
    params = [{'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}]
    best_svc_linear, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(svc_linear, params, X_train, X_test, 
                                                                                   y_train, y_test, acc_lst, f1_lst, 
                                                                                   best_params_lst, cv_err_lst)
    svc_rbf = svm.SVC(kernel = 'rbf')
    params = [{'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0],
              'C': [40, 50, 60, 70, 80, 90]}]
    best_svc_rbf, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(svc_rbf, params, X_train, X_test, 
                                                                                y_train, y_test, acc_lst, f1_lst, 
                                                                                best_params_lst, cv_err_lst)
    svc_poly = svm.SVC(kernel = 'poly')
    params = [{'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0], 
              'C': [0.001, 0.01, 0.1, 1.0, 2.0, 5.0]}]
    best_svc_poly, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV(svc_poly, params, X_train, X_test, 
                                                                                 y_train, y_test, acc_lst, f1_lst, 
                                                                                 best_params_lst, cv_err_lst)
    best_models = [dtree_best_model, rf_best_model, xg_best_model, cb_best_model, lda_best_model, 
                   qda_best_model, logist_best_model, best_svc_linear, best_svc_rbf, best_svc_poly]
    models_names = ['Desicion Tree', 'Random Forest', 'XGBoost', 'Catboost', 'LDA', 'QDA', 'Logistic regression', 
                    'SVC (linear kernel)', 'SVC (rbf kernel)', 'SVC (poly kernel)']
    
    CL_models = open('{0}/CL_models.txt'.format(indicator_folder), 'w')
    for i in range(len(models_names)):
        text = str(models_names[i]) + '\nbest params: ' + str(best_params_lst[i]) + '\nCV error = ' + str(cv_err_lst[i]) + '\nAccuracy = ' + str(acc_lst[i]) + '\nf1 = ' + str(f1_lst[i]) + '\n\n'
        CL_models.write(text)
    CL_models.close()
    
    confusion_matrices(X, y, best_models, models_names, cognitive_indicator, indicator_folder)
    
    best_model_ind = acc_lst.index(max(acc_lst))
    si.SHAP_interpretation(best_models[best_model_ind], models_names[best_model_ind], X, correlation_frame, indicator_folder)