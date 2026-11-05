import hydra
import numpy as np
import random
import os
import mlflow
from functools import partial
from typing import Dict
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from utils.log import log_hyperparams, write_scores
from utils.dataset import load_dataset, balance_dataset
import pandas as pd
from utils.preproc import TSControllerTranslationCoordinateTransform

@hydra.main(version_base = None, config_path='./.conf',config_name='test')
def main(
    cfg: DictConfig
):
    # dataset
    # preprocessing    
    # pipeline
    # metrics
    # grid search CV
    # handle class imbalance   
    # 
    random.seed = cfg.random.seed
    np.random.seed(seed = cfg.random.seed)

    X_train, y_train = load_dataset(loader = cfg.dataset.dataset_loader,
                        path_to_dataset_dir = cfg.dataset.path_to_dataset_dir,
                        feature_names = cfg.dataset.feature_names,
                        index_file = cfg.dataset.train_index_file if 'train_index_file' in cfg.dataset else None,
                        gesture_names = cfg.dataset.gesture_names,
                        handedness = cfg.dataset.handedness if 'handedness' in cfg.dataset else None)                        

    if 'balanced' in cfg.dataset and cfg.dataset.balanced:
        X_train, y_train = balance_dataset(X_train, y_train)

    X_test, y_test = load_dataset(loader = cfg.dataset.dataset_loader,
                        path_to_dataset_dir = cfg.dataset.path_to_dataset_dir,
                        feature_names = cfg.dataset.feature_names,
                        index_file = cfg.dataset.test_index_file if 'test_index_file' in cfg.dataset else None,
                        gesture_names = cfg.dataset.gesture_names,
                        handedness = cfg.dataset.handedness if 'handedness' in cfg.dataset else None)                        
    
    preproc = [instantiate(
        component_class
    ) for _, component_class in cfg.preprocess.items()]

    classifier = instantiate(cfg.classifier)

    ########
    # DONT USE ClassifierPipeline. It messes up with dimensions of panel pd.DataFrame
    # it is not supposed to work with multivariate timeseries nor with series of different length !!!
    # clf = ClassifierPipeline(classifier, preproc)
    ######################
    
    clf = make_pipeline(*(preproc+[classifier]))
    clf.fit(X = X_train, y = y_train)    

    score_funcs = {
        'accuracy' : accuracy_score,
        'balanced_accuracy' : balanced_accuracy_score,
        'f1_score' : partial(f1_score, average='macro'),
        'precision_score' : partial(precision_score, average='macro'),
        'recall_score' : partial(recall_score, average='macro')
    }

    y_pred = clf.predict(X_test)
    scores = { }
    for score_name, score_func in score_funcs.items():
        scores["test_"+score_name] = (score_func(y_true=y_test, y_pred = y_pred),)

    write_scores(scores, os.path.join(HydraConfig.get().runtime.output_dir,'test_scores.csv'))    
    
    #
    # logging    
    #
    experiment_name = classifier.__class__.__name__
    mlflow.set_tracking_uri(cfg.mlflow.mlruns_path)
    exp = mlflow.set_experiment(experiment_name)    
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.set_tag("dataset",cfg.dataset.name)
        mlflow.set_tag("source","test")
        log_hyperparams(cfg)
        mlflow.log_metrics({k : float(np.mean(v)) for k, v in scores.items() })   
        mlflow.sklearn.log_model(clf,"model")

if __name__ == "__main__":
    main()