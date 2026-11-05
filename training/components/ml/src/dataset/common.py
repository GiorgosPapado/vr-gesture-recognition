import pandas as pd
import numpy as np
import msgpack
import os
import csv
from typing import List, Any, Optional, Tuple, Union
from glob import glob
from operator import itemgetter
from sklearn.model_selection import train_test_split

def generate_train_test_split_index_file(
    path_to_dataset_dir : str, 
    out_path_to_train_split_txt : str,
    out_path_to_test_split_txt : str,
    gesture_names : Optional[str] = None,
    train_size : float = 0.7,
    random_state : int = 0
):
    dataset_files = [ ]
    dataset_gesture_labels = [ ]
    dataset_gesture_names = [ ]
    for filename in glob(path_to_dataset_dir+"/*.bin"):
        with open(filename,'rb') as f:
            sample_data = msgpack.load(f)
            sample_gesture_name = sample_data['GestureName']

            # check if we should import this gesture.
            import_gesture = gesture_names is None or sample_gesture_name in gesture_names
            if import_gesture:
                sample_label = sample_data['GestureID']
                base_filename = os.path.basename(filename)

                dataset_files.append(base_filename)
                dataset_gesture_labels.append(sample_label)
                dataset_gesture_names.append(sample_gesture_name)

    files_train, files_test, labels_train, labels_test, gestures_train, gestures_test = \
        train_test_split(dataset_files, dataset_gesture_labels, dataset_gesture_names, train_size = train_size, random_state=random_state, stratify=dataset_gesture_labels)

    # write train files index file
    with open(out_path_to_train_split_txt, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','gesture_id','gesture_name'])
        for row in zip(files_train, labels_train, gestures_train):
            writer.writerow(row)
    # write test files index file
    with open(out_path_to_test_split_txt, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','gesture_id','gesture_name'])
        for row in zip(files_test, labels_test, gestures_test):
            writer.writerow(row)    
