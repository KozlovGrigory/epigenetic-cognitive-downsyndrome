import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
import catboost as cb
from sklearn import tree, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def quality_metrics (y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = 'macro')
    return acc, f1

def Grid_Search_CV_CL(model, params, X_train, X_test, 
                      y_train, y_test, acc_lst, f1_lst, 
                      best_params_lst, cv_err_lst, y_pred_lst):
    grid_cv = GridSearchCV(estimator = model, param_grid = params, cv = 5, n_jobs = -1)
    grid_cv.fit(X_train, y_train)
    best_model, best_params, cv_err = grid_cv.best_estimator_, grid_cv.best_params_, grid_cv.best_score_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc, f1 = quality_metrics (y_test, y_pred)
    acc_lst.append(acc)
    f1_lst.append(f1)
    best_params_lst.append(best_params)
    cv_err_lst.append(cv_err)
    y_pred_lst.append(y_pred)
    return best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst

def confusion_matrices (X, y, best_models, models_names, cognitive_indicator, directory_folder):  
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

        cf_matrix = confusion_matrix(act_targets, pred_targets)
        ax = sns.heatmap(cf_matrix, annot = True, cmap = 'coolwarm')
        ax.set_title('{0}\nAccuracy = {1:0.3f}, f1 = {2:0.3f}'.format(models_names[best_models.index(model_opt)], acc_m, f1_m), fontsize = 12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.xaxis.set_ticklabels(['Small','Medium', 'Large'])
        ax.yaxis.set_ticklabels(['Small','Medium', 'Large'])
        
        directory_folder_ML = '{0}/ML_CL'.format(directory_folder)
        if not os.path.isdir(directory_folder_ML):
            os.mkdir(directory_folder_ML)
        
        plt.savefig('{0}/{1}_{2}_cl.png'.format(directory_folder_ML, cognitive_indicator, models_names[best_models.index(model_opt)]), dpi = 300)
        plt.close()
    return directory_folder_ML

def shap_interpretation (model, model_name, X, X_train, directory_folder_ML):
    shap.initjs()
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X)
    
    directory_folder_SHAP = '{0}/SHAP'.format(directory_folder_ML)
    if not os.path.isdir(directory_folder_SHAP):
            os.mkdir(directory_folder_SHAP)

    fig = plt.gcf()
    shap.summary_plot(shap_values, features = X, feature_names = X.columns, max_display = 50)
    fig.savefig('{0}/{1}_shap_beeswarm.png'.format(directory_folder_SHAP, model_name),
                format = 'png', dpi = 300, bbox_inches = 'tight')
    fig = plt.gcf()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names = X.columns)
    fig.savefig('{0}/{1}_shap_waterfall_legacy.png'.format(directory_folder_SHAP, model_name), 
                format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.close()
    pass

def machine_learning_CL (correlation_cpgs, indicator, cognitive_indicator, directory_folder):
    X = correlation_cpgs
    y = indicator['class']
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 111)
    best_params_lst, cv_err_lst, y_pred_lst = [], [], []
    acc_lst, f1_lst = [], []
    
    dtree_model = tree.DecisionTreeClassifier(random_state = 111)
    params = {'max_depth': range (1, 4, 1)}
    dtree_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(dtree_model, params, X_train, X_test, 
                                                                                       y_train, y_test, acc_lst, f1_lst, 
                                                                                       best_params_lst, cv_err_lst, y_pred_lst)
    rf_model = RandomForestClassifier(random_state = 111)
    params = {'n_estimators': range(10, 110, 10)}
    rf_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(rf_model, params, X_train, X_test, 
                                                                                    y_train, y_test, acc_lst, f1_lst, 
                                                                                    best_params_lst, cv_err_lst, y_pred_lst)
    xg_model = xgb.XGBClassifier(objective ='multi:softprob', use_label_encoder = False, random_state = 111)
    params = {'max_depth': range (1, 4, 1), 'n_estimators': range(10, 110, 10), 
              'learning_rate': [0.1, 0.01, 0.05]}
    xgb.set_config(verbosity = 0)
    xg_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(xg_model, params, X_train, X_test, 
                                                                                    y_train, y_test, acc_lst, f1_lst, 
                                                                                    best_params_lst, cv_err_lst, y_pred_lst)
    cb_model = cb.CatBoostClassifier(loss_function = 'MultiClass', random_state = 111)
    params = {'iterations': [100, 150, 200], 'learning_rate': [0.03, 0.1], 
              'depth': [2, 3, 4], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    cb_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(cb_model, params, X_train, X_test, 
                                                                                    y_train, y_test, acc_lst, f1_lst, 
                                                                                    best_params_lst, cv_err_lst, y_pred_lst)
    lda_model = LDA()
    params = {'solver': ['svd', 'lsqr', 'eigen']}
    lda_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(lda_model, params, X_train, X_test, 
                                                                                     y_train, y_test, acc_lst, f1_lst, 
                                                                                     best_params_lst, cv_err_lst, y_pred_lst)
    qda_model = QDA()
    params = {'reg_param': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
    qda_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(qda_model, params, X_train, X_test, 
                                                                                     y_train, y_test, acc_lst, f1_lst, 
                                                                                     best_params_lst, cv_err_lst, y_pred_lst)
    logist = LogisticRegression()
    params = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
    logist_best_model, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(logist, params, X_train, X_test, 
                                                                                        y_train, y_test, acc_lst, f1_lst, 
                                                                                        best_params_lst, cv_err_lst, y_pred_lst)
    svc_linear = svm.SVC(kernel = 'linear')
    params = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}
    best_svc_linear, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(svc_linear, params, X_train, X_test, 
                                                                                      y_train, y_test, acc_lst, f1_lst, 
                                                                                      best_params_lst, cv_err_lst, y_pred_lst)
    svc_rbf = svm.SVC(kernel = 'rbf')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0],
              'C': [40, 50, 60, 70, 80, 90]}
    best_svc_rbf, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(svc_rbf, params, X_train, X_test, 
                                                                                   y_train, y_test, acc_lst, f1_lst, 
                                                                                   best_params_lst, cv_err_lst, y_pred_lst)
    svc_poly = svm.SVC(kernel = 'poly')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0], 
              'C': [0.001, 0.01, 0.1, 1.0, 2.0, 5.0]}
    best_svc_poly, best_params_lst, cv_err_lst, acc_lst, f1_lst = Grid_Search_CV_CL(svc_poly, params, X_train, X_test, 
                                                                                    y_train, y_test, acc_lst, f1_lst, 
                                                                                    best_params_lst, cv_err_lst, y_pred_lst)
    
    best_models = [dtree_best_model, rf_best_model, xg_best_model, cb_best_model, lda_best_model, 
                   qda_best_model, logist_best_model, best_svc_linear, best_svc_rbf, best_svc_poly]
    models_names = ['Desicion Tree', 'Random Forest', 'XGBoost', 'Catboost', 'LDA', 'QDA', 'Logistic regression', 
                    'SVC (linear kernel)', 'SVC (rbf kernel)', 'SVC (poly kernel)']
    directory_folder_ML = error_analysis_graphs(X, y, best_models, models_names, cognitive_indicator, directory_folder)
    CL_models = open('{0}/{1}_CL_models.txt'.format(directory_folder_ML, cognitive_indicator), 'w')
    for i in range(0, len(models_names)):
        text = str(models_names[i]) + '\nbest params: ' + str(best_params_lst[i]) + '\nCV error = ' + str(cv_err_lst[i]) + '\nMAE = ' + str(MAE_lst[i]) + '\nr2 = ' + str(r2_lst[i]) + '\n\n'
        CL_models.write(text)
        shap_interpretation(best_models[i], models_names[i], X, X_train, directory_folder_ML)
    CL_models.close()
    return CL_models, directory_folder_ML