import pickle
import hydra
import numpy as np
import random
import os
from functools import partial
from typing import Dict
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from utils.log import write_scores
from utils.dataset import load_dataset, balance_dataset
from export.onnx import export as export_onnx
from functools import partial
import torch.onnx as onnx
import onnxruntime as ort
from tqdm import tqdm
import torch
import numpy as np

def evaluate_onnx(cfg, onnx_file_path: str):
    X_test, y_test = load_dataset(loader = cfg.dataset.dataset_loader,
                        path_to_dataset_dir = cfg.dataset.path_to_dataset_dir,
                        feature_names = cfg.dataset.feature_names,
                        index_file = cfg.dataset.test_index_file if 'test_index_file' in cfg.dataset else None,
                        gesture_names = cfg.dataset.gesture_names,
                        handedness = cfg.dataset.handedness if 'handedness' in cfg.dataset else None)                        
    
    onnx_model = ort.InferenceSession(onnx_file_path)

    class_map = { class_index : class_id for class_index, class_id in enumerate(np.unique(y_test.values))}

    y_pred = [ ]
    for i in tqdm(range(X_test.shape[0]), desc = 'Evaluating ONNX ...'):
        X = torch.from_numpy(np.vstack([X_test.iloc[i,:][idx] for idx in range(X_test.shape[1])])).unsqueeze(0)

        Y = torch.from_numpy(onnx_model.run(None, {'input': X.float().numpy()})[0])
        Y_pred = torch.argmax(Y,dim=1).item()
        y_pred.append(class_map[Y_pred])

    score_funcs = {
        'accuracy' : accuracy_score,
        'balanced_accuracy' : balanced_accuracy_score,
        'f1_score' : partial(f1_score, average='macro'),
        'precision_score' : partial(precision_score, average='macro'),
        'recall_score' : partial(recall_score, average='macro')
    }
    y_pred = np.array(y_pred)
    
    scores = { }
    for score_name, score_func in score_funcs.items():
        scores["test_"+score_name] = (score_func(y_true=y_test, y_pred = y_pred),)

    write_scores(scores, os.path.join(HydraConfig.get().runtime.output_dir,'test_scores_onnx.csv')) 

@hydra.main(version_base = None, config_path='./.conf',config_name='onnx')
def main(
    cfg: DictConfig
):   
 
    actions = {
        'to_onnx':          export_onnx,
        'evaluate_onnx':    evaluate_onnx
    }

    for action , kwargs in cfg.actions.items():
        actions[action](cfg, **kwargs)

if __name__ == "__main__":
    main()