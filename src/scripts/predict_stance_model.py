import argparse
from common.loadData import load_data
from model.roberta.roberta_model import predict_task
import pandas as pd
from common.score import scorePredict


def main(parser):
    args = parser.parse_args()
    test_set = args.test_set
    use_cuda = args.use_cuda
    model_dir_1_stage = args.model_dir_1_stage
    model_dir_2_stage = args.model_dir_2_stage
    features_1_stage = args.features_1_stage


    label_map = {'unrelated': 3, 'agree': 0, 'disagree': 1, 'discuss': 2}
    df_test = load_data(test_set, features_1_stage, label_map, 'test', '')

    if model_dir_1_stage != '':
        y_predict_1 = predict_task(df_test, use_cuda, model_dir_1_stage, len(features_1_stage))
        df_result = df_test
        df_result['predict'] = y_predict_1
        if model_dir_2_stage != '':
            df_y_1 = pd.DataFrame(y_predict_1, columns=['predict'])
            df_y_1_0 = df_y_1[df_y_1['predict'] == 0]
            df_y_1_1 = df_y_1[df_y_1['predict'] == 1]

            p_test_1 = df_test.loc[df_y_1_0.index]
            p_test_1['predict'] = df_y_1_0['predict'].values
            p_test_1['predict'] = p_test_1['predict'].replace(0, 3)

            df_test_2 = df_test.loc[df_y_1_1.index]
            y_predict_2 = predict_task(df_test_2, use_cuda, model_dir_2_stage, 0)
            df_test_2['predict'] = y_predict_2
            df_result = pd.concat([p_test_1, df_test_2], axis=0)

    labels = list(df_test['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(df_result['predict'].values, df_result['labels'].values, labels)
    print(result)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--test_set",
                        default="/data/FNC_PLM_originDataset_test_all_summary_v2.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir_1_stage",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of the model first stage to predict.")

    parser.add_argument("--model_dir_2_stage",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of the model second stage to predict.")

    parser.add_argument("--features_1_stage",
                        default=[],
                        nargs='+',
                        help="This parameter is features of model first stage for predict.")

    main(parser)