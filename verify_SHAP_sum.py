from utils.load_shap_values import SHAPLoader
from utils.load_models import ModelLoader
from utils.load_civilcomments import CivilCommentsLoader

def sum_SHAP_values(SHAP_values, sample_id, toxic=True):
    if toxic:
        column = 'SHAP_toxic'
    else:
        column = 'SHAP_not_toxic'
    samples = SHAP_values[SHAP_values['id'] == sample_id]
    sum = samples[column].sum()
    return sum

def get_avg_SHAP(SHAP_values, toxic=True):
    sum = 0
    index = 0
    if toxic:
        column = 'SHAP_toxic'
    else:
        column = 'SHAP_not_toxic'
    while index < SHAP_values.shape[0]:
        sum += SHAP_values.iloc[index, SHAP_values.columns.get_loc(column)]
        index += 1
    return sum/SHAP_values.shape[0]

def get_sample_prediction(pipeline, text):
    return pipeline.predict(text)

if __name__ == '__main__':
    # Fields to update to run an experiment
    erm_SHAP_input_data_filename = 'data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = 'data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = 'data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = 'data/cnc_shap_values.txt'
    sample_ids = [5501204, 745527, 6011650]
    data_filename = 'data/all_data_with_identities.csv'
    erm_model_checkpoint_filename = 'data/civilcomments_erm_early.pth.tar'
    cnc_model_checkpoint_filename = 'data/civilcomments_cnc_pretrained.pth.tar'


    erm_SHAP_values = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).SHAP_values
    cnc_SHAP_values = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).SHAP_values

    erm_avg_SHAP = get_avg_SHAP(erm_SHAP_values)
    cnc_avg_SHAP = get_avg_SHAP(cnc_SHAP_values)

    cnc_pipeline = ModelLoader(cnc_model_checkpoint_filename).pipeline
    erm_pipeline = ModelLoader(erm_model_checkpoint_filename).pipeline
    data = CivilCommentsLoader(data_filename).data

    print('--------- ERM MODEL SCORES ----------')
    for sample_id in sample_ids:
        sum_sample_1 = sum_SHAP_values(erm_SHAP_values, sample_id)
        prediction = get_sample_prediction(erm_pipeline, data.loc[data["id"] == sample_id, 'text'].item())
        print('---- SAMPLE ID ' + str(sample_id) + '-----')
        print('---- SAMPLE SHAP SUM -----')
        print(sum_sample_1)
        print('---- SAMPLE TOXIC PREDICTION SCORE -----')
        print(prediction[0][1]['score'])
        print('---- SAMPLE PREDICTION MINUS SAMPLE SHAP SUM (EXPECTED TO BE THE SAME FOR ALL SAMPLES) -----')
        print(prediction[0][1]['score'] - sum_sample_1)
        print('')

    print('--------- CNC MODEL SCORES ----------')
    for sample_id in sample_ids:
        sum_sample_1 = sum_SHAP_values(cnc_SHAP_values, sample_id)
        prediction = get_sample_prediction(cnc_pipeline, data.loc[data["id"] == sample_id, 'text'].item())
        print('---- SAMPLE ID ' + str(sample_id) + '-----')
        print('---- SAMPLE SHAP SUM -----')
        print(sum_sample_1)
        print('---- SAMPLE TOXIC PREDICTION SCORE -----')
        print(prediction[0][1]['score'])
        print('---- SAMPLE PREDICTION MINUS SAMPLE SHAP SUM (EXPECTED TO BE THE SAME FOR ALL SAMPLES) -----')
        print(prediction[0][1]['score'] - sum_sample_1)
        print('')
