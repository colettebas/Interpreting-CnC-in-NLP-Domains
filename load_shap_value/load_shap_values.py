from ast import literal_eval
import pandas as pd

def decode_tokens(token_array_file):
    with open(token_array_file, 'rb') as f:
        tokens = f.read()
    tokens = tokens.decode("ISO-8859-1")
    tokens = tokens.split('array')
    tokens.pop(0)
    tokens_list = []
    for i in range(len(tokens)):
        tokens[i] = tokens[i][1::]
        array_end_index = tokens[i].index('],')
        tokens[i] = tokens[i][:array_end_index+2:]
        tokens_list = tokens_list + literal_eval(tokens[i])[0]
    return tokens_list

def decode_shap_values(values_array_file):
    with open(values_array_file, 'rb') as f:
        values = f.read()
    values = values.decode("ISO-8859-1")
    values = values.split('array')
    values.pop(0)
    values_list = []
    for i in range(len(values)):
        values[i] = values[i][1::]
        array_end_index = values[i].index(')')
        values[i] = values[i][:array_end_index:]
        values_list = values_list + literal_eval(values[i])
    return values_list

def convert_shap_files(values_file, tokens_file):
    tokens = decode_tokens(tokens_file)
    values = decode_shap_values(values_file)
    shap_values = pd.DataFrame(values, columns=['SHAP_not_toxic', 'SHAP_toxic'])
    shap_values['token'] = tokens
    return shap_values

if __name__ == '__main__':
    SHAP_values = convert_shap_files('erm_shap_values.txt', 'erm_shap_data.txt')