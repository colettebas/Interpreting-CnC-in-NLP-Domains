from utils.load_shap_values import SHAPLoader
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Fields to update to run an experiment
    erm_SHAP_input_data_filename = 'data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = 'data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = 'data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = 'data/cnc_shap_values.txt'
    tokens = ['male', 'man', 'boy', 'he', 'him', 'his']

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
    erm_SHAP_values = erm_SHAP_values.loc[erm_SHAP_values['token'].isin(tokens)]
    erm_SHAP_values['Demographic'] = 'Male'
    erm_SHAP_values = erm_SHAP_values.drop(['SHAP_not_toxic', 'id', 'token', 'abs_SHAP_toxic'], axis=1)

    cnc_SHAP_values = cnc_SHAP_values.rename(columns={"SHAP_toxic": "SHAP Value"})
    cnc_SHAP_values['Model'] = 'CnC'
    cnc_SHAP_values = cnc_SHAP_values.loc[cnc_SHAP_values['token'].isin(tokens)]
    cnc_SHAP_values['Demographic'] = 'Male'
    cnc_SHAP_values = cnc_SHAP_values.drop(['SHAP_not_toxic', 'id', 'token'], axis=1)

    combined_SHAP_values = erm_SHAP_values.append(cnc_SHAP_values, ignore_index=True)
    print(combined_SHAP_values)

    sns.set_theme(style="whitegrid")

    # Draw violin plot with raw values
    sns.violinplot(data=combined_SHAP_values, x="Demographic", y="SHAP Value", hue="Model",
                   split=True, inner="quart", linewidth=1,
                   palette={"ERM": "b", "CnC": ".85"})
    sns.despine(left=True)
    plt.savefig('sample_violin.png')

    # Draw violin plot with normalized values
    plt.clf()
    sns.violinplot(data=combined_SHAP_values, x="Demographic", y="Normalized SHAP Value", hue="Model",
                   split=True, inner="quart", linewidth=1,
                   palette={"ERM": "b", "CnC": ".85"})
    sns.despine(left=True)
    plt.savefig('normalized_sample_violin.png')