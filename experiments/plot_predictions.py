from utils.load_shap_values import SHAPLoader
from utils import load_models as lm
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parent_dir = os.path.abspath(os.getcwd())
    erm_model = lm.ModelLoader(parent_dir + '/data/civilcomments_erm_early.pth.tar').pipeline
    cnc_model = lm.ModelLoader(parent_dir + '/data/civilcomments_cnc_pretrained.pth.tar').pipeline

    erm_SHAP_input_data_filename = parent_dir + '/data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = parent_dir + '/data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = parent_dir + '/data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = parent_dir + '/data/cnc_shap_values.txt'
        
    erm_texts_list = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).texts_list
    cnc_texts_list = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).texts_list

    erm_predictions = erm_model.predict(erm_texts_list)
    cnc_predictions = cnc_model.predict(cnc_texts_list)

    erm_predictions = sorted([x[1]['score'] for x in erm_predictions])
    cnc_predictions = sorted([x[1]['score'] for x in cnc_predictions])

    plt.scatter(range(len(erm_predictions)), erm_predictions, label='ERM')
    plt.scatter(range(len(cnc_predictions)), cnc_predictions, label='CNC')
    plt.ylabel('P(y_hat = 1) i.e toxic prediction')
    plt.xlabel('index')
    plt.title('Comparision of distribution P(y_hat = 1) for CNC and ERM')
    plt.legend()
    plt.show()
    plt.savefig('model_predictions.png')