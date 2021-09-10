# HeadlineStanceChecker

**(2021/01/10) create readme **

This repository contains the code of the main experiments of our research using the Fake News Challenge dataset (FNC). The task consists of classifying a given headline with respect to its body-text (agree, disagree, discuss, and unrelated). 
Our approach is based on a neural network that uses automatic summaries generated from the body-text, so it is recommended to execute the code in a GPU device.

The code to train the models together with the generated summaries are available. In addition, our pre-trained models are also available. Please follow the instructions below, considering that some differences in the results could be due to library versions and the features of the devices where the code is executed.

You can use this code in your host environment or in a docker container.
### Requirements
* Python 3.6
* Pytorch >= 1.6
* Transformers >= 4.6.1
* Linux OS or Docker

### Installation in your host
If you use Linux based on Debian or Ubuntu, you can execute this script to install all requirements:
```bash 
    sudo ./install_host.sh
```

#### Manual installation

Create a Python Environment and activate it:
```bash 
    virtualenv roberta --python=python3
    cd ./roberta
    source bin/activate
```
Install the required dependencies. 
You need to have at least version 21.0.1 of pip installed. Next you may install requirements.txt.

```bash
pip3 install --upgrade pip
pip3 install torch==1.8.1
pip3 install -r requirements.txt
```

Install this requirement to train in GPU
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


Download the FNC-dataset and the generatic summaries from this link:
```bash
wget -O data.zip "https://drive.google.com/uc?export=download&id=1AF-U0jjud1eJaKmWVxNBLvQdpxp6LEj3"
unzip data.zip
rm data.zip
```

If you want to predict with our models you will download this folder.
Download pre-training models from this google account:

```bash
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id=1Ob-CVMlfBBRhcBpG1-iaPP_g9SZYjW2s' -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O models.zip \
     'https://docs.google.com/uc?export=download&id=1Ob-CVMlfBBRhcBpG1-iaPP_g9SZYjW2s&confirm='$(<confirm.txt)
unzip models.zip
rm models.zip
rm confirm.txt
rm cookies.txt
```
### Installation in docker container
You only need to have docker installed. 
 
Docker images tested:
* pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
* pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
* pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

You need to grant permission to install_docker.sh file:
```bash
chmod 777 install_docker.sh
```

If you have GPU you will use this command:
```bash
docker run --name name_container -it --net=host --gpus device=device_number -v folder_dir_with_code:/workspace pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash -c "./install_docker.sh"
```

If you have not GPU you will use this command:
```bash
docker run --name name_container -it --net=host -v folder_dir_with_code:/workspace pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash -c "./install_docker.sh"
```

These commands should be executed into the code folder.
### Description of scripts

##### Train and predict models
If you want train and predict your model you will use train_predict_model.py

These parameters allow to configure the system to train or predict.

|Field|Description|
|---|---|
|type_classify|This parameter is used to choose the type of classifier (stance, related and all).|
|use_cuda|This parameter can be used if cuda is present.|
|training_set|This parameter is the relative directory of the training set.|
|test_set|This parameter is the relative directory of the test set.|
|model_dir|This parameter is the relative directory of the model for prediction.|
|features_1_stage|This parameter contains the features of the model for the first stage of prediction (cosineSimilarity, max_score_in_position, overlap, spacySimilarity, jaccardScore, hellingerScore, kullback_leiblerScore).|


For example, if you want to train and predict "stance" as the type of classifier:
```bash
--type_classify 'stance'
```
For example, if you want to train and predict "related" as the type of classifier with different features:
```bash
--type_classify 'related' --features_1_stage 'cosineSimilarity' 'max_score_in_position' 'overlap'
```

Execute this command to train and predict "related" as the type of classifier with different features.

```bash
PYTHONPATH=src python src/scripts/train_predict_model.py --training_set "/data/FNC_PLM_originDataset_train_all_summary_v2.json" --test_set "/data/FNC_PLM_originDataset_test_all_summary_v2.json" --type_classify 'related' --features_1_stage 'cosineSimilarity' 'max_score_in_position' 'overlap' --use_cuda
```

Execute this command to predict "related" as the type of classifier with different features using our pre-trained model
```bash
PYTHONPATH=src python src/scripts/train_predict_model.py --model_dir "/model" --test_set "/data/FNC_PLM_originDataset_test_all_summary_v2.json" --type_classify 'related' --features_1_stage 'cosineSimilarity' 'max_score_in_position' 'overlap' --use_cuda
```

##### Predicting using the whole architecture (HeadlineStanceChecker)  
If you want to predict all models, you will use predict_stance_model.py

These parameters allow to configure the system to obtain the prediction with one stage (HeadlineStanceChecker-1stage) or with two stages (HeadlineStanceChecker-2stage).

|Field|Description|
|---|---|
|use_cuda|This parameter can be used if cuda is present.|
|test_set|This parameter is the relative directory of the test set.|
|model_dir_1_stage|This parameter is the relative directory of the model for predicting the first stage, i.e., related and unrelated.|
|model_dir_2_stage|This parameter is the relative directory of the model for predicting the second stage, i.e., agree, disagree, and discuss.|
|features_1_stage|This parameter contains the features of the model for the first stage of prediction (cosineSimilarity, max_score_in_position, overlap, spacySimilarity, jaccardScore, hellingerScore, kullback_leiblerScore).|

Execute this command to predict the FNC classes with your models 
```bash
PYTHONPATH=src python src/scripts/predict_stance_model.py --model_dir_1_stage "/model_1" --model_dir_2_stage "/model_2" --test_set "/data/FNC_PLM_originDataset_test_all_summary_v2.json" --features_1_stage 'cosineSimilarity' 'max_score_in_position' 'overlap' --use_cuda
```
Execute this command to predict the FNC classes with our provided pre-trained models
```bash
PYTHONPATH=src python src/scripts/predict_stance_model.py --model_dir_1_stage "/models/related" --model_dir_2_stage "/models/stance" --test_set "/data/FNC_PLM_originDataset_test_all_summary_v2.json" --features_1_stage 'cosineSimilarity' 'max_score_in_position' 'overlap' --use_cuda
```

Note: If you don't have GPU remove "--use_cuda" in the commands

### Contacts:
If you have any questions please contact the authors.   
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com 
  
### License:
  * Apache License Version 2.0 