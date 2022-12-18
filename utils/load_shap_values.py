from ast import literal_eval
import pandas as pd

class SHAPLoader():

    def __init__(self, input_data_filename: str, input_values_filename: str):
        self.SHAP_values = self.convert_shap_files(input_data_filename, input_values_filename)
        self.texts_list = self.decode_text(input_data_filename)

    def decode_tokens(self, token_array_file):
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
    
    def decode_text(self, token_array_file):
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
            tokens_list.append(''.join(literal_eval(tokens[i])[0]))
        return tokens_list

    def decode_shap_values(self, values_array_file, sample_indices):
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
            tuples = literal_eval(values[i])
            for tuple in tuples:
                tuple.append(sample_indices[i])
            values_list = values_list + tuples
        return values_list

    def convert_shap_files(self, data_file, values_file):
        sample_indices = [5501204,  745527, 6011650, 6028377,  689468, 5836741, 5325575, 5268943,
                          7068476,  395975, 7017781,  374886, 5996752, 5790334,  874205, 5247322,
                          7087935,  551485,  837864, 5225928, 5996362, 6049191, 7010178,  969205,
                          912424, 5807085, 5258444,  302244, 5691616, 5706291,  263826, 5673529,
                          1042100, 5668535, 5660666, 5671172, 5849196,  854766, 5714403, 5677939,
                          317738, 5661906, 7030152, 5273685, 5667538, 5271747, 5634407, 6195881,
                          5250160, 5413617, 5082363, 6289685,  765032, 5223210, 5619106, 5227845,
                          6212185,  805615, 5926602, 5739778,  562695,  916552, 5299060, 7121989,
                          5061615, 5413135, 5619134, 5636688, 6207105, 5355613,  459284, 5481493,
                          5437664, 6194010, 6086468, 5021733, 5670747,  364360, 5701054, 6232221,
                          5702495, 6137334, 7186383, 5654569, 5738177, 5866509,  345804, 6000535,
                          5801799, 7037625,  591666,  703019, 1058652, 5437859, 5758811, 1036211,
                          5482323,  796796, 5130262, 5150596,  871928, 6217765, 5740264,  291905,
                          332398,  940570, 1009617,  478245, 5670871, 5615730, 5760844,  528144,
                          7098717, 5754496, 7166976, 1014032, 7068738, 5722790, 6081854, 5617979,
                          911525,  924296, 5972210, 5060992, 6091732,  924059, 5177923,  359103,
                          5363060, 5104459, 6102821, 5015383, 5996226, 7157963,  929809, 5150270,
                          5807595,  830440, 5405057,  793671, 6010742, 5668823, 6211944, 6042510,
                          6180079, 5246452, 1065278, 7091716, 5838669, 5535379, 7100054, 5511312,
                          6156879, 5789327, 4982134, 6192361,  518302, 5952937, 5665505,  982496,
                          5146957, 5720920, 5376205, 6229436, 5870940, 5507166, 5395067,  768060,
                          5567718,  775551, 6198500, 6078953, 7158473, 6305426,  440051, 7026882,
                          850396,  929241, 7038266, 5449238,  688431, 7087713, 6160565, 1070663,
                          7067877, 5791746, 5358390, 5316820, 5662524, 5547323, 6150904,  634605,
                          277145, 5988370, 6068702, 5083110, 5864967, 6264813, 5570643, 1063971,
                          5229899, 6170349, 6221978, 6222995, 5798216, 7064045, 5827378, 6203736,
                          691407, 5319080, 6318572,  778195, 5202048, 5050173, 1058174, 5064416,
                          6093098,  468354,  640894,  683294, 1021911, 5633642, 6132423, 5903169,
                          841347, 6091704,  603228, 5864320, 6305041, 1036874, 5971755,  865220,
                          5548384,  816903, 1022930, 6279524,  901438, 7139278, 1033869, 7071539,
                          5697223,  884837, 5154939, 5208818, 7047243, 6186959, 1083160, 5863209,
                          1079446,  635378, 5448718, 5172545, 5802195,  448084, 7069276,  349702,
                          508013,  891563, 7187074, 7156493, 5885610, 5890739, 5146285, 5577310,
                          264044, 5659300,  486934, 7068576, 6197915, 6008986, 6309507, 5205561,
                          810498,  509950, 5310217, 6221773,  245711, 6081457, 5666897, 5848243,
                          1001118, 5662096,  735476, 7008599, 5082780]
        tokens = self.decode_tokens(data_file)
        values = self.decode_shap_values(values_file, sample_indices)
        shap_values = pd.DataFrame(values, columns=['SHAP_not_toxic', 'SHAP_toxic', 'id'])
        shap_values['token'] = tokens
        return shap_values