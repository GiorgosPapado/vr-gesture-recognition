import numpy as np
import pandas as pd
from dataset.controller import load_dataset as load_controller_dataset
from dataset.hands import load_dataset as load_hand_dataset
from typing import Optional, List, Tuple

def load_dataset(loader: str,
                **dataset_loader_kwargs):
    """    
    :dataset_loader_kwargs: optional keyword arguments to pass to the dataset loader, eg in order to consider only gestures of specified handedness
    """
    dataset_loader = {
        'controller': load_controller_dataset,
        'hand': load_hand_dataset
    }

    X, y = dataset_loader[loader](
            **dataset_loader_kwargs
    )
    return X, y


def balance_dataset(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    X: pandas DataFrame of dataset instances
    y: np ndarray of class labels one for each instance in X
    :returns: Tuple[pd.DataFrame,np.ndarray] a new dataset that is balanced, created by repetitions of random samples of each class to match the number of samples of majority class
    """

    class_labels = np.unique(y)
    class_count  = [int(np.sum(y == label)) for label in class_labels]
    target_count = max(class_count)
    expand_X = [ ]
    expand_Y = [ ]
    for label, label_count in zip(class_labels,class_count):
        selector = y == label
        more = target_count - label_count
        if more > 0:
            expand_X.append(X.loc[selector].sample(more, replace=True).copy())            
            expand_Y.append(np.array([label] * more, dtype=int))

    newX = pd.concat([X] + expand_X, ignore_index=True, verify_integrity=True)
    newY = np.concatenate((y,*expand_Y))

    return newX, newY
