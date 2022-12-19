from utils.load_shap_values import SHAPLoader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Fields to update to run an experiment
    erm_SHAP_input_data_filename = 'data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = 'data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = 'data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = 'data/cnc_shap_values.txt'
    sample_id = 5060992
    #5619134 demographic: gay
    #7026882 demographic: other religion
    #7068476 demographic: man
    #5060992 demographic: Muslim
    #5807085 demographic: women, white, men
    #5791746 demographic: black
    #7067877 demographic: black, white
    #6093098 demographic: black, white

    erm_SHAP_values = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).SHAP_values
    cnc_SHAP_values = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).SHAP_values

    # filter for SHAP values for only this sample
    erm_SHAP_values = erm_SHAP_values.loc[erm_SHAP_values['id'] == sample_id]
    cnc_SHAP_values = cnc_SHAP_values.loc[cnc_SHAP_values['id'] == sample_id]

    # Calculate the normalized SHAP values by dividing them by the sum of the
    # absolute value of the SHAP values.  We need to take the absolute value because
    # if we do not then the SHAP values could be larger than the sum.
    erm_SHAP_values['abs_SHAP_toxic'] = erm_SHAP_values['SHAP_toxic'].abs()
    cnc_SHAP_values['abs_SHAP_toxic'] = cnc_SHAP_values['SHAP_toxic'].abs()

    erm_sum_SHAP = erm_SHAP_values['abs_SHAP_toxic'].sum()
    cnc_sum_SHAP = cnc_SHAP_values['abs_SHAP_toxic'].sum()

    erm_SHAP_values['SHAP_toxic_normalized'] = erm_SHAP_values['SHAP_toxic']/erm_sum_SHAP
    cnc_SHAP_values['SHAP_toxic_normalized'] = cnc_SHAP_values['SHAP_toxic']/cnc_sum_SHAP

    # Plot raw SHAP values
    y_pos = np.arange(len(cnc_SHAP_values['token']))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(y_pos - width/2, erm_SHAP_values['SHAP_toxic'], width, label='ERM')
    rects2 = ax.barh(y_pos + width/2, cnc_SHAP_values['SHAP_toxic'], width, label='CnC')

    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Value by Model for Sample ' + str(sample_id))
    ax.legend()
    ax.set_yticks(y_pos,)
    ax.set_yticklabels(cnc_SHAP_values['token'])
    ax.invert_yaxis()  # labels read top-to-bottom

    fig.tight_layout()
    plt.grid(axis = 'y')

    plt.savefig(str(sample_id) + '_SHAP_comparison.png')
    plt.show()

    # Plot Normalized SHAP values

    fig, ax = plt.subplots()
    rects1 = ax.barh(y_pos - width/2, erm_SHAP_values['SHAP_toxic_normalized'], width, label='ERM')
    rects2 = ax.barh(y_pos + width/2, cnc_SHAP_values['SHAP_toxic_normalized'], width, label='CnC')

    ax.set_xlabel('Normalized SHAP Value')
    ax.set_title('Normalized SHAP Value by Model for Sample ' + str(sample_id))
    ax.legend()
    ax.set_yticks(y_pos,)
    ax.set_yticklabels(cnc_SHAP_values['token'])
    ax.invert_yaxis()  # labels read top-to-bottom

    fig.tight_layout()
    plt.grid(axis = 'y')

    plt.savefig(str(sample_id) + '_SHAP_normalized_comparison.png')
    plt.show()