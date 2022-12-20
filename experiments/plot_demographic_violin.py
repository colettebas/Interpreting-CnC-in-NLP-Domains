from utils.load_shap_values import SHAPLoader
from utils.load_words import WordsLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
    # Fields to update to run an experiment
    parent_dir = os.path.abspath(os.getcwd())
    erm_SHAP_input_data_filename = parent_dir + '/data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = parent_dir + '/data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = parent_dir + '/data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = parent_dir + '/data/cnc_shap_values.txt'
    demography_words_filenames = {
        'race': parent_dir + '/data/race_words_reddit.csv',
        'religion': parent_dir + '/data/religion_words_reddit.csv',
        'sexuality': parent_dir + '/data/sexuality_words_reddit.csv'
    }
    demography_words = WordsLoader(demography_words_filenames).words

    erm_SHAP_values = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).SHAP_values
    cnc_SHAP_values = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).SHAP_values

    # Calculate the normalized SHAP values by dividing them by the sum of the
    # absolute value of the SHAP values.  We need to take the absolute value because
    # if we do not then the SHAP values could be larger than the sum.
    erm_SHAP_values['abs_SHAP_toxic'] = erm_SHAP_values['SHAP_toxic'].abs()
    cnc_SHAP_values['abs_SHAP_toxic'] = cnc_SHAP_values['SHAP_toxic'].abs()

    erm_sum_SHAP = erm_SHAP_values['abs_SHAP_toxic'].sum()
    cnc_sum_SHAP = cnc_SHAP_values['abs_SHAP_toxic'].sum()

    erm_SHAP_values['Normalized SHAP Value'] = erm_SHAP_values['SHAP_toxic']/erm_sum_SHAP
    cnc_SHAP_values['Normalized SHAP Value'] = cnc_SHAP_values['SHAP_toxic']/cnc_sum_SHAP

    erm_SHAP_values = erm_SHAP_values.rename(columns={"SHAP_toxic": "SHAP Value"})
    erm_SHAP_values['Model'] = 'ERM'
    cnc_SHAP_values = cnc_SHAP_values.rename(columns={"SHAP_toxic": "SHAP Value"})
    cnc_SHAP_values['Model'] = 'CnC'

    erm_SHAP_values['Demographic'] = 'None'
    cnc_SHAP_values['Demographic'] = 'None'
    for demograpghy, tokens in demography_words.items():
        erm_SHAP_values.iloc[erm_SHAP_values['token'].isin(tokens), -1] = demograpghy
        cnc_SHAP_values.iloc[cnc_SHAP_values['token'].isin(tokens), -1] = demograpghy

    erm_SHAP_values = erm_SHAP_values[erm_SHAP_values['Demographic']!='None']
    cnc_SHAP_values = cnc_SHAP_values[cnc_SHAP_values['Demographic']!='None']

    combined_SHAP_values = pd.concat([erm_SHAP_values, cnc_SHAP_values], axis=0, ignore_index=True)
    print(combined_SHAP_values)

    sns.set_theme(style="whitegrid")

    # Draw violin plot with raw values
    sns.violinplot(data=combined_SHAP_values, x="Demographic", y="SHAP Value", hue="Model",
                   split=True, inner="quart", linewidth=1,
                   palette={"ERM": "b", "CnC": ".85"}).set_title('SHAP values for demographic words')
    sns.despine(left=True)
    plt.show()
    plt.savefig('sample_violin.png')

    # Draw violin plot with normalized values
    plt.clf()
    sns.violinplot(data=combined_SHAP_values, x="Demographic", y="Normalized SHAP Value", hue="Model",
                   split=True, inner="quart", linewidth=1,
                   palette={"ERM": "b", "CnC": ".85"}).set_title('Normalized SHAP values for demographic words')
    sns.despine(left=True)
    plt.show()
    plt.savefig('normalized_sample_violin.png')