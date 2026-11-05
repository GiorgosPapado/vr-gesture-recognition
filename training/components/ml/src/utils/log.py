import mlflow
import csv
from omegaconf import DictConfig
from typing import Dict

def log_hyperparams(cfg : DictConfig):
    
    for key, value in cfg.classifier.items():
        mlflow.log_param("classifier.hparam."+key, value)
    
    for component_name in cfg.preprocess:
        for param_name, param_value in cfg.preprocess[component_name].items():
            mlflow.log_param(component_name+"."+param_name,param_value)


def write_scores(scores : Dict[str,float], scores_filepath : str):
    with open(scores_filepath,'w', newline='') as f:
        writer = csv.writer(f)
        for score_name, score_values in filter(lambda x: "test_" in x[0], scores.items()):
            writer.writerow([score_name, *map(lambda x: "{:2.4f}".format(x) ,score_values)])