from utils.load_shap_values import SHAPLoader
from utils.load_models import ModelLoader
from utils.load_civilcomments import CivilCommentsLoader
import shap

def sum_SHAP_values(SHAP_values, sample_id, toxic=True):
    if toxic:
        column = 'SHAP_toxic'
    else:
        column = 'SHAP_not_toxic'
    samples = SHAP_values[SHAP_values['id'] == sample_id]
    sum = samples[column].sum()
    return sum

def get_sample_prediction(pipeline, text):
    return pipeline.predict(text)

def get_base_prediction(sample_id, model, data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data.loc[data["id"] == sample_id, 'text'])
    return shap_values.base_values[0][1]


if __name__ == '__main__':
    # Fields to update to run an experiment
    erm_SHAP_input_data_filename = 'data/erm_shap_data.txt'
    erm_SHAP_input_values_filename = 'data/erm_shap_values.txt'
    cnc_SHAP_input_data_filename = 'data/cnc_shap_data.txt'
    cnc_SHAP_input_values_filename = 'data/cnc_shap_values.txt'
    sample_ids = [6102821, 7026882]
    data_filename = 'data/all_data_with_identities.csv'
    erm_model_checkpoint_filename = 'data/civilcomments_erm_early.pth.tar'
    cnc_model_checkpoint_filename = 'data/civilcomments_cnc_pretrained.pth.tar'

    data = CivilCommentsLoader(data_filename).data

    erm_SHAP_values = SHAPLoader(erm_SHAP_input_data_filename, erm_SHAP_input_values_filename).SHAP_values
    cnc_SHAP_values = SHAPLoader(cnc_SHAP_input_data_filename, cnc_SHAP_input_values_filename).SHAP_values

    erm_pipeline = ModelLoader(erm_model_checkpoint_filename).pipeline
    cnc_pipeline = ModelLoader(cnc_model_checkpoint_filename).pipeline

    print('--------- ERM MODEL SCORES ----------')
    for sample_id in sample_ids:
        sum_sample_1 = sum_SHAP_values(erm_SHAP_values, sample_id)
        base_prediction = get_base_prediction(sample_id, erm_pipeline, data)
        prediction = get_sample_prediction(erm_pipeline, data.loc[data["id"] == sample_id, 'text'].item())
        print('---- SAMPLE ID ' + str(sample_id) + '-----')
        print('---- SAMPLE SHAP SUM -----')
        print(sum_sample_1)
        print('---- SAMPLE TOXIC PREDICTION SCORE -----')
        print(prediction[0][1]['score'])
        print('---- DIFFERENCE: SAMPLE PREDICTION MINUS SAMPLE SHAP SUM (EXPECTED TO BE THE SAME FOR SAMPLES WITH SAME # TOKENS) -----')
        print(prediction[0][1]['score'] - sum_sample_1)
        print('---- BASE PREDICTION (EXPECTED TO BE THE SAME AS DIFFERENCE) -----')
        print(base_prediction)
        print('')

    print('--------- CNC MODEL SCORES ----------')
    for sample_id in sample_ids:
        sum_sample_1 = sum_SHAP_values(cnc_SHAP_values, sample_id)
        base_prediction = get_base_prediction(sample_id, cnc_pipeline, data)
        prediction = get_sample_prediction(cnc_pipeline, data.loc[data["id"] == sample_id, 'text'].item())
        print('---- SAMPLE ID ' + str(sample_id) + '-----')
        print('---- SAMPLE SHAP SUM -----')
        print(sum_sample_1)
        print('---- SAMPLE TOXIC PREDICTION SCORE -----')
        print(prediction[0][1]['score'])
        print('---- DIFFERENCE: SAMPLE PREDICTION MINUS SAMPLE SHAP SUM (EXPECTED TO BE THE SAME FOR SAMPLES WITH SAME # TOKENS) -----')
        print(prediction[0][1]['score'] - sum_sample_1)
        print('---- BASE PREDICTION (EXPECTED TO BE THE SAME AS DIFFERENCE) -----')
        print(base_prediction)
        print('')
