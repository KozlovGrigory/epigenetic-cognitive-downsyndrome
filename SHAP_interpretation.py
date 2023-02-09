import os
import shap
import matplotlib.pyplot as plt

def SHAP_interpretation (model, model_name, X, correlation_frame, indicator_folder):
    indicator_folder_SHAP = '{0}/SHAP'.format(indicator_folder)
    if not os.path.isdir(indicator_folder_SHAP):
            os.mkdir(indicator_folder_SHAP)
    
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)        
    
    x, y = 15, 12
    fig_inch = (x/2.54, y/2.54)
    fig = plt.figure()
    shap.summary_plot(shap_values, features = X, feature_names = correlation_frame.index, show = False)
    plt.gcf().set_size_inches(fig_inch)
    fig.patch.set_facecolor('white')
    plt.title(model_name)
    fig.savefig('{0}/beeswarm.png'.format(indicator_folder_SHAP), format = 'png', dpi = 300, bbox_inches = 'tight')

    for patient_id in len(shap_values):
        fig = plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[patient_id], 
                                               feature_names = correlation_frame.index, show = False)
        plt.gcf().set_size_inches(fig_inch)
        fig.patch.set_facecolor('white')
        fig.savefig('{0}/waterfall_legacy_{1}.png'.format(indicator_folder_SHAP, patient_id), 
                        format = 'png', dpi = 300, bbox_inches = 'tight')