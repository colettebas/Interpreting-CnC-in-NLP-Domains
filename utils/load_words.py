# from ast import literal_eval
import pandas as pd

class WordsLoader():

    def __init__(self, words_filenames: dict):
        self.words = self.convert_words_files(words_filenames)

    def convert_words_files(self, words_filenames):
        output = {}
        for key, val in words_filenames.items():
            words_df = pd.read_csv(val)
            if 'valid' in words_df.columns:
                words_df = words_df[words_df['valid'] == 1]
            output[key] = list(set(list(words_df['word'])))
        return output