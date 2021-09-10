import argparse
from common.loadData import load_data
from model.roberta.roberta_model import train_predict_model, predict_task
import pandas as pd
from common.score import scorePredict


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    test_set = args.test_set
    use_cuda = args.use_cuda
    model_dir = args.model_dir
    type_classify = args.type_classify
    features_1_stage = args.features_1_stage

    features = features_1_stage
    if type_classify == 'related':
        label_map = {'unrelated': 0, 'agree': 1, 'disagree': 1, 'discuss': 1}
    elif type_classify == 'stance':
        label_map = {'agree': 0, 'disagree': 1, 'discuss': 2}
        features = []
    else:
        label_map = {'unrelated': 3, 'agree': 0, 'disagree': 1, 'discuss': 2}

    df_train = load_data(training_set, features, label_map, 'training', type_classify)
    df_test = load_data(test_set, features, label_map, 'test', type_classify)

    if model_dir == '':
        _, y_predict = train_predict_model(df_train, df_test, True, use_cuda, len(features))
    else:
        y_predict = predict_task(df_test, use_cuda, model_dir, len(features))

    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(y_predict, labels_test, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--type_classify",
                        default="",
                        type=str,
                        help="This parameter is used for choose type of classify.")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--training_set",
                        default="/data/FNC_PLM_originDataset_train_all_summary_v2.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--test_set",
                        default="/data/FNC_PLM_originDataset_test_all_summary_v2.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--features_1_stage",
                        default=[],
                        nargs='+',
                        help="This parameter is the features of model first stage for predict.")

    main(parser)