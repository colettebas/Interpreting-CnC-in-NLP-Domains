from utils.load_shap_values import SHAPLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

erm_SHAP_input_data_filename = 'data/erm_shap_data.txt'
erm_SHAP_input_values_filename = 'data/erm_shap_values.txt'
cnc_SHAP_input_data_filename = 'data/cnc_shap_data.txt'
cnc_SHAP_input_values_filename = 'data/cnc_shap_values.txt'
sample_id = 6093098
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

erm_SHAP_values['abs_SHAP_toxic'] = erm_SHAP_values['SHAP_toxic'].abs()
cnc_SHAP_values['abs_SHAP_toxic'] = cnc_SHAP_values['SHAP_toxic'].abs()

diff_SHAP_values = cnc_SHAP_values['abs_SHAP_toxic'] - erm_SHAP_values['abs_SHAP_toxic']
diff_SHAP_values_df = pd.DataFrame(list(zip(diff_SHAP_values, cnc_SHAP_values['token'])), columns=['diff_shap_vals', 'tokens'])
diff_SHAP_values_df = diff_SHAP_values_df.sort_values(by=['diff_shap_vals'], ascending=False)
diff_SHAP_values = diff_SHAP_values_df['diff_shap_vals'][:10]
diff_SHAP_tokens = diff_SHAP_values_df['tokens'][:10]

plt.bar(diff_SHAP_tokens, diff_SHAP_values, color ='maroon', width = 0.4)
 
plt.ylabel("Difference in SHAP values")
plt.xlabel("Tokens")
plt.title("Difference in ERM and CNC model SHAP values")
plt.show()