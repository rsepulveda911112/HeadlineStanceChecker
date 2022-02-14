import argparse
import os
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from common.preprocessing.TF_IDF_cosineSimilarity import word_feature
from common.preprocessing.sumy_summary import TextRank_Summarizer
from common.preprocessing.similarityMetrics import SimilarityMetrics
from common.preprocessing.util import combine
from common.preprocessing.overlap import word_overlap_feature
from common.reader import JSONLineReader
import tqdm
import json
import time


def main(parser):
    args = parser.parse_args()
    dataset_in = args.dataset_in
    dataset_out = args.dataset_out

    start_time = time.time()
    exec_preprocessing(dataset_in, dataset_out)
    print(time.time() - start_time)

    

def exec_preprocessing(dataset_in, dataset_out):
    jsonlReader = JSONLineReader()
    datas = jsonlReader.read(os.getcwd() + dataset_in)
    df = pd.DataFrame(datas)
    tr = TextRank_Summarizer('english', 5)
    n_threads = cpu_count() - 4
    pool = Pool(n_threads)
    output = []
    # df = df.iloc[0:100]
    
    with Pool(n_threads) as p:
        output = list(tqdm.tqdm(p.imap(tr.cal_summary, df['sentences2']), total=len(df)))
    df = df.drop(columns=['sentences2'])
    df['sentences2'] = output
    
    headlines_lemas = []
    bodies_lemas = []
    heads_bodies_combine = []

    with Pool(n_threads) as p:
        proccess = p.imap(combine, df.itertuples(name=None), chunksize=5)
        for head_body_combine, headline_lema, body_lema in tqdm.tqdm(proccess, total=len(df)):
            headlines_lemas.append(headline_lema)
            bodies_lemas.append(body_lema)
            heads_bodies_combine.append(head_body_combine)

    f = open(os.getcwd() + dataset_out, "w+")
    similarityMetrics = SimilarityMetrics('/resource/model_lda_preprocess', heads_bodies_combine)
   
    with Pool(n_threads) as p:
        values = p.starmap(cal_metrics, tqdm.tqdm([(row, headline_lema, body_lema, similarityMetrics)
                                        for (index, row), headline_lema, body_lema
                                        in zip(df.iterrows(), headlines_lemas, bodies_lemas)]))

    for row in tqdm.tqdm(values, total=len(values)):
        f.write(json.dumps(row) + "\n")



def cal_metrics(row, headline_lema, summary_lema, similarityMetrics):
    headline = row['sentence1']
    summary = row['sentences2']
    overlap_feature = word_overlap_feature(headline, summary, False)
    tf_idf, cosine_similarity_value = word_feature(headline, summary)    
    hellingerScore, jaccardScore, kullbackScore = similarityMetrics.allMetrics(headline_lema, summary_lema)
    kullbackScore = np.float64(kullbackScore)
    return {'Id_Article': row['Id_Article'], 'sentence1': headline,
                            'sentences2': summary, 'label': row['label'],
                            'cosineSimilarity': cosine_similarity_value,
                            'jaccardScore': jaccardScore, 'hellingerScore': hellingerScore,
                            'kullback_leiblerScore': kullbackScore, 'overlap': overlap_feature}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--dataset_in",
                        default="/data/FNC_body_train.json",
                        type=str,
                        help="This parameter is the relative dir of dataset.")

    parser.add_argument("--dataset_out",
                        default="/data/FNC_body_train_features.json",
                        type=str,
                        help="This parameter is the relative dir of preprocessed dataset.")

    main(parser)