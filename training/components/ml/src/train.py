import hydra
import numpy as np
import random
import csv
import os
import mlflow
import warnings
from typing import Dict
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.model_selection import cross_validate
#from sktime.classification.compose._pipeline import ClassifierPipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, make_scorer
from utils.log import log_hyperparams, write_scores
from utils.dataset import load_dataset, balance_dataset

@hydra.main(version_base = None, config_path='./.conf',config_name='train')
def main(
    cfg: DictConfig
):
    # dataset
    # preprocessing    
    # pipeline
    # metrics
    # grid search CV
    # handle class imbalance

    random.seed(cfg.random.seed)
    np.random.seed(seed = cfg.random.seed)

    X, y = load_dataset(loader=cfg.dataset.dataset_loader, 
                    path_to_dataset_dir = cfg.dataset.path_to_dataset_dir,
                    feature_names = cfg.dataset.feature_names,
                    index_file = cfg.dataset.train_index_file,
                    gesture_names = cfg.dataset.gesture_names,
                    handedness = cfg.dataset.handedness if 'handedness' in cfg.dataset else None)

    if 'balanced' in cfg.dataset and cfg.dataset.balanced:
        X, y = balance_dataset(X, y)

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
    # clf.fit(X = X, y = y)   # don't fit. cross_validate will do. 

    scorecb = {
        'accuracy' : make_scorer(accuracy_score),
        'balanced_accuracy' : make_scorer(balanced_accuracy_score),
        'f1_score' : make_scorer(f1_score, average='macro'),
        'precision_score' : make_scorer(precision_score, average='macro'),
        'recall_score' : make_scorer(recall_score, average='macro')
    }
    try:
        scores = cross_validate(
            estimator = clf,
            X = X,
            y = y,
            cv = 5,
            n_jobs=5,
            scoring = scorecb
        )

        write_scores(scores, os.path.join(HydraConfig.get().runtime.output_dir,'cross_val_scores.csv'))
        final_evaluation_score = float(np.mean(scores[f'test_{cfg.hyopt.monitor_metric}']))
        
        #
        # logging    
        #
        experiment_name = classifier.__class__.__name__
        mlflow.set_tracking_uri(cfg.mlflow.mlruns_path)
        exp = mlflow.set_experiment(experiment_name)    
        with mlflow.start_run(experiment_id=exp.experiment_id):
            mlflow.set_tag("dataset",cfg.dataset.name)
            mlflow.set_tag("source","train")
            log_hyperparams(cfg)
            mlflow.log_metrics({k : float(np.mean(v)) for k, v in scores.items() })
            # don't log clf here. the fitted estimators are computed inside cross_validate
            # mlflow.sklearn.log_model(clf,"model")

    except Exception as ex:        
        warnings.warn(f"FIT FAILED. Exception {str(ex)}")
        final_evaluation_score = -np.inf

    return final_evaluation_score


if __name__ == "__main__":
    main()