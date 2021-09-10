import os
from common.reader import JSONLineReader
import pandas as pd
from tqdm import tqdm


def load_data(file, feature_name, map_value, type_dataset, type_classify):
    jsonlReader = JSONLineReader()
    datas = jsonlReader.read(os.getcwd() + file)
    df_in = pd.DataFrame(datas)
    headlines = []
    bodies = []
    labels = []
    real_labels = []
    features = []
    if type_classify == 'stance':
        df_in = df_in[df_in['label'] != 'unrelated']
    for index, row in tqdm(df_in.iterrows(), total=len(df_in), desc=f"Load {type_dataset} set"):
        headlines.append(row['sentence1'])
        body_text = ''
        for value in row['sentences2']:
           body_text += value + ' '
        bodies.append(body_text)
        labels.append(map_value[row['label']])
        real_labels.append(row['label'])
        list_feature = []
        for value in feature_name:
            list_feature.append(row[value])
        features.append(list_feature)
    if len(features[0]) == 0:
        features = [0] * len(df_in)

    list_of_tuples = list(zip(headlines, bodies, labels, features))
    df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels', 'features'])
    print(df['labels'].value_counts())
    return df