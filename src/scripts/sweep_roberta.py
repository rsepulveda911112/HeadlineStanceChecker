import wandb
from numpy import random
from src.scripts.train_predict_model import exec_model


def train():
    #Add wandb project
    wandb.init(config=wandb.config, project="")    
    exec_model("/data/FNC_PLM_originDataset_train_all_summary_v2.json",
                   "/data/FNC_PLM_originDataset_test_all_summary_v2.json",
                   True, '', 'related', ['cosineSimilarity', 'max_score_in_position', 'overlap'], [], wandb.config)

train()