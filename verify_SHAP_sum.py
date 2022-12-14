from utils.load_shap_values import SHAPLoader
from utils.load_models import ModelLoader
from utils.load_civilcomments import CivilCommentsLoader

def sum_SHAP_values(SHAP_values, start_index, end_index, toxic=True):
    index = start_index
    sum = 0
    if toxic:
        column = 'SHAP_toxic'
    else:
        column = 'SHAP_not_toxic'
    while index <= end_index:
        sum += SHAP_values.iloc[index, SHAP_values.columns.get_loc(column)]
        index += 1
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
    SHAP_input_data_filename = 'data/erm_shap_data.txt'
    SHAP_input_values_filename = 'data/erm_shap_values.txt'
    sample_start_index = 0
    sample_end_index = 202
    sample_id = 5501204
    data_filename = 'data/all_data_with_identities.csv'
    model_checkpoint_filename = 'data/civilcomments_erm_early.pth.tar'


    SHAP_values = SHAPLoader(SHAP_input_data_filename, SHAP_input_values_filename).SHAP_values
    sum_sample_1 = sum_SHAP_values(SHAP_values, sample_start_index, sample_end_index)  ## SAMPLE ID 5501204, CNCDIFF .004568, ERM 0.110963
    #sum_sample_1 = sum_SHAP_values(SHAP_values, 203, 301) ##SAMPLE ID 745527, CNCDIFF .004042, ERM 0.161709
    #sum_sample_1 = sum_SHAP_values(SHAP_values, 302, 428) ##SAMPLE ID 6011650, CNCDIFF .003815, ERMDIFF 0.08442

    avg_SHAP = get_avg_SHAP(SHAP_values)
    #print(SHAP_values.iloc[302:430].to_string())

    #cnc_pipeline = ModelLoader('data/civilcomments_cnc_pretrained.pth.tar').pipeline
    erm_pipeline = ModelLoader(model_checkpoint_filename).pipeline
    data = CivilCommentsLoader(data_filename).data
    prediction = get_sample_prediction(erm_pipeline, data.loc[data["id"] == sample_id, 'text'].item())
    print(data.loc[data["id"] == sample_id, 'text'].item())
    print('---- SAMPLE SHAP SUM -----')
    print(sum_sample_1)
    print('---- AVERAGE SHAP VALUE -----')
    print(avg_SHAP)
    print('---- SAMPLE TOXIC PREDICTION SCORE -----')
    print(prediction[0][1]['score'])
    print('---- SAMPLE PREDICTION MINUS SAMPLE SUM -----')
    print(prediction[0][1]['score'] - sum_sample_1)
