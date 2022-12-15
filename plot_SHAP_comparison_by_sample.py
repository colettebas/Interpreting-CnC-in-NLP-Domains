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

    erm_SHAP_values = erm_SHAP_values.loc[erm_SHAP_values['id'] == sample_id]
    cnc_SHAP_values = cnc_SHAP_values.loc[cnc_SHAP_values['id'] == sample_id]

    y_pos = np.arange(len(cnc_SHAP_values['token']))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(y_pos - width/2, erm_SHAP_values['SHAP_toxic'], width, label='ERM')
    rects2 = ax.barh(y_pos + width/2, cnc_SHAP_values['SHAP_toxic'], width, label='CnC')

    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Value by Model for Sample ' + str(sample_id))
    ax.legend()
    ax.set_yticks(y_pos, labels=cnc_SHAP_values['token'])
    ax.invert_yaxis()  # labels read top-to-bottom

    fig.tight_layout()

    plt.show()
    plt.savefig(str(sample_id) + '_SHAP_comparison.png')