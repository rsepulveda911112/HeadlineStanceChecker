import os
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model.out_simple_transformer.ClassificationModel import ClassificationModel


def train_predict_model(df_train, df_test, is_predict, is_evaluate ,use_cuda, value_head, wandb_project=None, wandb_config=None):
    df_train = df_train.sample(frac=1)
    labels = list(df_train['labels'].unique())
    labels.sort()

    if wandb_project:
        if wandb_config != None:
            wandb.init(config=wandb_config, project=wandb_project)
        else:
            wandb.init(config=wandb.config, project=wandb_project)
        wandb_config = wandb.config
    else:
        wandb_config = dict()

    df_eval = None
    if is_evaluate:
        df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)

    model = ClassificationModel('roberta', 'roberta-large',
             num_labels=len(labels), use_cuda=use_cuda, args={
            'learning_rate': 1e-5, 'num_train_epochs': 3,
            'reprocess_input_data': True, 'overwrite_output_dir': True,
            'process_count': 10, 'train_batch_size': 2,
            'eval_batch_size': 2, 'max_seq_length': 512,
            'evaluate_during_training': is_evaluate,
            'multiprocessing_chunksize': 500, 'fp16': True,
            'fp16_opt_level': 'O1', 'value_head': value_head,
            'wandb_project': 'wandb_project',
            'tensorboard_dir': 'tensorboard'})
    # sweep_config = wandb.config

    model.train_model(df_train, eval_df=df_eval)

    results = ''
    if is_predict:
        text_a = df_test['text_a']
        text_b = df_test['text_b']
        feature = df_test['features']
        df_result = pd.concat([text_a, text_b, feature], axis=1)
        value_in = df_result.values.tolist()
        y_predict, model_outputs_test = model.predict(value_in)
    else:
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']

    y_predict = np.argmax(model_outputs_test, axis=1)
    return results, y_predict


def predict_task(df_test, use_cuda, model_dir, value_head):
    model = ClassificationModel(model_type='roberta', model_name=os.getcwd() + model_dir, use_cuda=use_cuda,
                                args={'value_head': value_head})
    text_a = df_test['text_a']
    text_b = df_test['text_b']
    feature = df_test['features']
    df_result = pd.concat([text_a, text_b, feature], axis=1)
    value_in = df_result.values.tolist()
    y_predict, model_outputs_test = model.predict(value_in)
    y_predict = np.argmax(model_outputs_test, axis=1)
    return y_predict