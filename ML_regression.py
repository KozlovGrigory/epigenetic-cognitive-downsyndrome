import os
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import catboost as cb
from sklearn import tree, svm, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression, LogisticRegression
from matplotlib.lines import Line2D 

def error_analysis_graphs(X, y, best_models, models_names, cognitive_indicator, indicator_folder, cognitive_frame):
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
        
        color_labels = cognitive_frame['sex'].unique()  
        rgb_values = [sns.color_palette("Set1", 12)[1], sns.color_palette("Set1", 12)[2]]
        color_map = dict(zip(color_labels, rgb_values))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(pred_targets, act_targets, c = cognitive_frame['sex'].map(color_map))  
        handles = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = v, label = k, markersize = 10) for k, v in color_map.items()]
        ax.legend(title = 'Sex:', handles = handles, bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 14, title_fontsize = 14)

        b = lambda a: a
        a = np.linspace(min(act_targets), max(act_targets), 2)
        plt.plot(a, b(a), color = '#E41A1C')
        MAE_m = metrics.mean_absolute_error(act_targets, pred_targets)
        r2_m = metrics.r2_score(act_targets, pred_targets)
        plt.title('{0}\nMAE = {1:0.3f}, r2 = {2:0.3f}'.format(models_names[best_models.index(model_opt)], MAE_m, r2_m), fontsize = 14)
        plt.xlabel('Predicted', fontsize = 14)
        plt.ylabel('Actual', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.tight_layout()
        plt.savefig('{0}/{1}_{2}_svg.svg'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), facecolor ="white", dpi = 300)
        plt.savefig('{0}/{1}_{2}_png.png'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), facecolor ="white", dpi = 300)
        plt.savefig('{0}/{1}_{2}_tif.tif'.format(indicator_folder, cognitive_indicator, models_names[best_models.index(model_opt)]), facecolor ="white", dpi = 300)   
        plt.close()
        
def quality_metrics (y_test, y_pred):
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    return MAE, r2

def Grid_Search_CV(model, params, X_train, X_test, 
                   y_train, y_test, MAE_lst, r2_lst, 
                   best_params_lst, cv_err_lst, y_pred_lst):
    grid_cv = GridSearchCV(estimator = model, param_grid = params, scoring = 'r2', cv = 5, n_jobs = -1)
    grid_cv.fit(X_train, y_train)
    y_pred = grid_cv.predict(X_test)
    best_params, cv_err = grid_cv.best_params_, grid_cv.best_score_
    
    MAE, r2 = quality_metrics (y_test, y_pred)
    MAE_lst.append(MAE)
    r2_lst.append(r2)
    best_params_lst.append(best_params)
    cv_err_lst.append(cv_err)

    return grid_cv, best_params_lst, cv_err_lst, MAE_lst, r2_lst

def ML_regression (correlation_cpgs, correlation_frame, methylation_frame, cognitive_frame, cognitive_indicator, indicator_folder):
    indicator_folder = '{0}/ML_regression'.format(indicator_folder)
    if not os.path.isdir(indicator_folder):
            os.mkdir(indicator_folder)
    
    X = methylation_frame[correlation_frame.index]
    y = indicator_classes['class']
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 111)
    best_params_lst, cv_err_lst = [], []
    MAE_lst, r2_lst = [], []
    
    dtree_model = tree.DecisionTreeRegressor(random_state = 111)
    params = {'max_depth': range (1, 4, 1)}
    dtree_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(dtree_model, params, X_train, X_test, 
                                                                                        y_train, y_test, MAE_lst, r2_lst, 
                                                                                        best_params_lst, cv_err_lst)
    rf_model = RandomForestRegressor(random_state = 111)
    params = {'n_estimators': range(10, 110, 10)}
    rf_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(rf_model, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst)
    xg_model = xgb.XGBRegressor(objective ='reg:squarederror', random_state = 111, use_label_encoder = False)
    params = {'max_depth': range (1, 4, 1), 'n_estimators': range(10, 110, 5), 'learning_rate': [0.001, 0.01, 0.1, 0.01, 0.05]}
    xgb.set_config(verbosity = 0)
    xg_best_model, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(xg_model, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst)
         
    cb_model = cb.CatBoostRegressor(loss_function = 'MAE', random_state = 111)
    params = {'iterations': [100, 150, 200, 300, 400], 'learning_rate': [0.01, 0.1, 0.2, 0.3], 
              'depth': [2, 3, 4], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    cb_best_model_MAE, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(cb_model, params, X_train, X_test, 
                                                                                         y_train, y_test, MAE_lst, r2_lst, 
                                                                                         best_params_lst, cv_err_lst)
    linear_model = LinearRegression()
    scores = []
    cvs = cross_val_score(estimator = linear_model, X = X, y = y, cv = 5, scoring = 'neg_mean_absolute_error')
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
                                                                                       best_params_lst, cv_err_lst)
    svr_rbf = svm.SVR(kernel = 'rbf')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 
              'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0], 
              'C': [40, 50, 60, 70, 80, 90]} 
    best_svr_rbf, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(svr_rbf, params, X_train, X_test, 
                                                                                    y_train, y_test, MAE_lst, r2_lst, 
                                                                                    best_params_lst, cv_err_lst)
    svr_poly = svm.SVR(kernel = 'poly')
    params = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0], 
              'coef0': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0],
              'C': [0.001, 0.01, 0.1, 1.0, 2.0, 5.0]} 
    best_svr_poly, best_params_lst, cv_err_lst, MAE_lst, r2_lst = Grid_Search_CV_REG(svr_poly, params, X_train, X_test, 
                                                                                     y_train, y_test, MAE_lst, r2_lst, 
                                                                                     best_params_lst, cv_err_lst)
    best_models = [dtree_best_model, rf_best_model, xg_best_model, cb_best_model_RMSE, cb_best_model_MAE, 
                   linear_model, best_svr_linear, best_svr_rbf, best_svr_poly]
    models_names = ['Desicion Tree', 'Random Forest', 'XGBoost', 'Catboost (RMSE)', 'Catboost (MAE)', 'Linear regression',
                    'SVR (linear kernel)', 'SVR (rbf kernel)', 'SVR (poly kernel)']
    
    REG_models = open('{0}/REG_models.txt'.format(indicator_folder_ML), 'w')
    for i in range(len(models_names)):
        text = str(models_names[i]) + '\nbest params: ' + str(best_params_lst[i]) + '\nCV error = ' + str(cv_err_lst[i]) + '\nMAE = ' + str(MAE_lst[i]) + '\nr2 = ' + str(r2_lst[i]) + '\n\n'
        REG_models_txt.write(text)
    REG_models_txt.close()
    
    error_analysis_graphs(X, y, best_models, models_names, cognitive_indicator, indicator_folder, cognitive_frame)
    
    best_model_ind = acc_lst.index(min(MAE_lst))
    si.SHAP_interpretation(best_models[best_model_ind], models_names[best_model_ind], X, correlation_frame, indicator_folder)