import pandas as pd

class CivilCommentsLoader():

    def __init__(self, data_filename):
        # load the civil comments dataset
        dataset = pd.read_csv(data_filename)

        #translate toxicity values to labels
        dataset.loc[dataset["toxicity"] >= 0.5, "toxicity"] = 1
        dataset.loc[dataset["toxicity"] < 0.5, "toxicity"] = 0

        self.data = pd.DataFrame({'id':dataset['id'], 'text':dataset['comment_text'],'toxic':dataset['toxicity']})
