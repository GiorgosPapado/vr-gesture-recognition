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
from export.factories.registry import get_op

def evaluate_model(cfg, **kwargs):
    X_test, y_test = load_dataset(loader = cfg.dataset.dataset_loader,
                        path_to_dataset_dir = cfg.dataset.path_to_dataset_dir,
                        feature_names = cfg.dataset.feature_names,
                        index_file = cfg.dataset.test_index_file if 'test_index_file' in cfg.dataset else None,
                        gesture_names = cfg.dataset.gesture_names,
                        handedness = cfg.dataset.handedness if 'handedness' in cfg.dataset else None)                        
    
    class_map = { class_index : class_id for class_index, class_id in enumerate(np.unique(y_test.values))}
    with open(r'C:\Users\giorgospap\Desktop\EKETA\INFINITY\vr-gestures\components\ml\src\mlruns\299390844902800384\279b07cd0e924c358f570ca9f721fcc6\artifacts\model\model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # X_test = X_test.iloc[83:85,:]
    y_pred = clf.predict(X_test)

    twin = get_op(clf)
    
    y_predt = [ ]
    for i in tqdm(range(X_test.shape[0]), desc = 'Evaluating ONNX ...'):
        X = torch.from_numpy(np.vstack([X_test.iloc[i,:][idx] for idx in range(X_test.shape[1])])).unsqueeze(0)

        Y = twin.forward(X)
        Y_pred = torch.argmax(Y,dim=1).item()
        y_predt.append(class_map[Y_pred])
    
    y_predt = np.array(y_predt)
    ydiff = np.argwhere(y_predt != y_pred)

    error_count = np.sum(y_predt != y_pred)
    print(f"Error count: {error_count}")



@hydra.main(version_base = None, config_path='./.conf',config_name='onnx')
def main(
    cfg: DictConfig
):   
    evaluate_model(cfg)
if __name__ == "__main__":
    main()