import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import catboost as cb
from sklearn import tree, svm, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression, LogisticRegression

def quality_metrics (y_test, y_pred):
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    return MAE, r2

def Grid_Search_CV_REG(model, params, X_train, X_test, 
                      y_train, y_test, MAE_lst, r2_lst, 
                      best_params_lst, cv_err_lst, y_pred_lst):
    grid_cv = GridSearchCV(estimator = model, param_grid = params, 
                           scoring = 'r2', cv = 5, n_jobs = -1)
    grid_cv.fit(X_train, y_train)
    best_model, best_params, cv_err = grid_cv.best_estimator_, grid_cv.best_params_, grid_cv.best_score_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    MAE, r2 = quality_metrics (y_test, y_pred)
    MAE_lst.append(MAE)
    r2_lst.append(r2)
    best_params_lst.append(best_params)
    cv_err_lst.append(cv_err)
    y_pred_lst.append(y_pred)
    return best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst

def error_analysis_graphs(X, y, best_models, models_names, cognitive_indicator, directory_folder):
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

        plt.figure(figsize = (5, 5))
        plt.plot(pred_targets, act_targets, "o", color = '#3b4cc0', alpha = 0.7)
        b = lambda a: a
        a = np.linspace(min(act_targets), max(act_targets), 2)
        plt.plot(a, b(a), color = 'red')
        MAE_m = metrics.mean_absolute_error(act_targets, pred_targets)
        r2_m = metrics.r2_score(act_targets, pred_targets)
        plt.title('{0}\nMAE = {1:0.3f}, r2 = {2:0.3f}'.format(models_names[best_models.index(model_opt)], MAE_m, r2_m), fontsize = 12)
        plt.xlabel('Predicted', fontsize = 12)
        plt.ylabel('Actual', fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        
        directory_folder_ML = '{0}/ML_REG'.format(directory_folder)
        if not os.path.isdir(directory_folder_ML):
            os.mkdir(directory_folder_ML)
        
        plt.savefig('{0}/{1}_{2}_reg.png'.format(directory_folder_ML, cognitive_indicator, models_names[best_models.index(model_opt)]), dpi = 300, bbox_inches = 'tight')
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

def machine_learning_REG (correlation_cpgs, cognitive_frame, cognitive_indicator, directory_folder):
    X = correlation_cpgs
    y = cognitive_frame[cognitive_indicator]
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 111)
    best_params_lst, cv_err_lst, y_pred_lst = [], [], []
    MAE_lst, r2_lst = [], []
    
    dtree_model = tree.DecisionTreeRegressor(random_state = 111)
    params = {'max_depth': range (1, 4, 1)}
    dtree_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(dtree_model, params, X_train, X_test, 
                                                                                        y_train, y_test, MAE_lst, r2_lst, 
                                                                                        best_params_lst, cv_err_lst, y_pred_lst)
    rf_model = RandomForestRegressor(random_state = 111)
    params = {'n_estimators': range(10, 110, 10)}
    rf_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(rf_model, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst, y_pred_lst)
    xg_model = xgb.XGBRegressor(objective ='reg:squarederror', random_state = 111, use_label_encoder = False)
    params = {'max_depth': range (1, 4, 1), 'n_estimators': range(10, 110, 5), 'learning_rate': [0.001, 0.01, 0.1, 0.01, 0.05]}
    xgb.set_config(verbosity = 0)
    xg_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(xg_model, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst, y_pred_lst)
    
    cb_model = cb.CatBoostRegressor(loss_function = 'RMSE', random_state = 111)
    params = {'iterations': [100, 150, 200, 300, 400], 'learning_rate': [0.01, 0.1, 0.2, 0.3], 
              'depth': [2, 3, 4], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    cb_best_model_RMSE, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(cb_model, params, X_train, X_test, 
                                                                                          y_train, y_test, MAE_lst, r2_lst, 
                                                                                          best_params_lst, cv_err_lst, y_pred_lst)      
    cb_model = cb.CatBoostRegressor(loss_function = 'MAE', random_state = 111)
    params = {'iterations': [100, 150, 200, 300, 400], 'learning_rate': [0.01, 0.1, 0.2, 0.3], 
              'depth': [2, 3, 4], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    cb_best_model_MAE, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(cb_model, params, X_train, X_test, 
                                                                                         y_train, y_test, MAE_lst, r2_lst, 
                                                                                         best_params_lst, cv_err_lst, y_pred_lst)
    linear_model = LinearRegression()
    scores = []
    cvs = cross_val_score(estimator = linear_model, X = X, y = y, 
                          cv = 5, scoring = 'neg_mean_absolute_error')
    scores.append(cvs)
    scores = np.array(scores)
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    MAE, r2 = quality_metrics (y_test, y_pred)
    best_params_lst.append('-')
    cv_err_lst.append(scores.mean())
    MAE_lst.append(MAE)
    r2_lst.append(r2)
    
    svr_linear = svm.SVR(kernel = 'linear')
    params = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]} 
    best_svr_linear, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(svr_linear, params, X_train, X_test, 
                                                                                       y_train, y_test, MAE_lst, r2_lst, 
                                                                                       best_params_lst, cv_err_lst, y_pred_lst)
    svr_rbf = svm.SVR(kernel = 'rbf')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 
              'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0], 
              'C': [40, 50, 60, 70, 80, 90]} 
    best_svr_rbf, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(svr_rbf, params, X_train, X_test, 
                                                                                    y_train, y_test, MAE_lst, r2_lst, 
                                                                                    best_params_lst, cv_err_lst, y_pred_lst)
    svr_poly = svm.SVR(kernel = 'poly')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 
              'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0],
              'C': [0.001, 0.01, 0.1, 1.0, 2.0, 5.0]} 
    best_svr_poly, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(svr_poly, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst, y_pred_lst)
    best_models = [dtree_best_model, rf_best_model, xg_best_model, cb_best_model_RMSE, cb_best_model_MAE, 
                   linear_model, best_svr_linear, best_svr_rbf, best_svr_poly]
    models_names = ['Desicion Tree', 'Random Forest', 'XGBoost', 'Catboost (RMSE)', 'Catboost (MAE)', 'Linear regression',
                    'SVR (linear kernel)', 'SVR (rbf kernel)', 'SVR (poly kernel)']
    directory_folder_ML = error_analysis_graphs(X, y, best_models, models_names, cognitive_indicator, directory_folder)
    REG_models_txt = open('{0}/{1}_REG_models.txt'.format(directory_folder_ML, cognitive_indicator), 'w')
    for i in range(len(models_names)):
        text = str(models_names[i]) + '\nbest params: ' + str(best_params_lst[i]) + '\nCV error = ' + str(cv_err_lst[i]) + '\nMAE = ' + str(MAE_lst[i]) + '\nr2 = ' + str(r2_lst[i]) + '\n\n'
        REG_models_txt.write(text)
        shap_interpretation(best_models[i], models_names[i], X, X_train, directory_folder_ML)
    REG_models_txt.close()
    return REG_models_txt, directory_folder_ML