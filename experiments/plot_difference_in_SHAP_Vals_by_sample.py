import matplotlib.pyplot as plt
import pandas as pd
import os

import sys

# setting path
sys.path.append('../Interpreting-CnC-in-NLP-Domains')
from utils.load_shap_values import SHAPLoader

if __name__ == '__main__':
    parent_dir = os.path.abspath(os.getcwd())
    erm_SHAP_input_data_filename = parent_dir + '/data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = parent_dir + '/data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = parent_dir + '/data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = parent_dir + '/data/cnc_shap_values.txt'

    # for sample_id in sample_ids:
    erm_SHAP_values = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).SHAP_values
    cnc_SHAP_values = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).SHAP_values

    erm_SHAP_values['SHAP_toxic'] = erm_SHAP_values['SHAP_toxic'] / sum(erm_SHAP_values['SHAP_toxic'].abs())
    cnc_SHAP_values['SHAP_toxic'] = cnc_SHAP_values['SHAP_toxic'] / sum(cnc_SHAP_values['SHAP_toxic'].abs())

    diff_SHAP_values = cnc_SHAP_values['SHAP_toxic'] - erm_SHAP_values['SHAP_toxic']
    diff_SHAP_values_df = pd.DataFrame(list(zip(diff_SHAP_values, cnc_SHAP_values['token'])), columns=['diff_shap_vals', 'tokens'])
    diff_SHAP_values_df = diff_SHAP_values_df.sort_values(by=['diff_shap_vals'], ascending=False)
    diff_SHAP_values = diff_SHAP_values_df['diff_shap_vals'][:30]
    diff_SHAP_tokens = diff_SHAP_values_df['tokens'][:30]

    fig = plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 14})

    colors = ['green'] * 30
    # colors[5] = 'red'
    # colors[8] = 'red'
    # colors[17] = 'red'
    # colors[23] = 'red'
    plt.bar(diff_SHAP_tokens, diff_SHAP_values, color = colors, width = 0.4)

    plt.ylabel("Differnce in normalized SHAP values (CNC - ERM)")
    plt.xlabel("Top 20 tokens")
    plt.xticks(rotation=90)
    plt.savefig('./diff_plot_normalized.png')
    plt.show()