import json
import pandas as pd
import numpy as np
import msgpack
import os
from csv import DictReader
from typing import List, Any, Optional, Tuple, Union
from glob import glob
from operator import itemgetter
from sklearn.model_selection import train_test_split


##
## Unity3D msgpack serializes Vector3 as (X,Y,Z) and Quaternion as (X,Y,Z,W)
##

#from random import randint, random, choices
# from sktime.datatypes._panel._convert import from_3d_numpy_to_nested, from_nested_to_3d_numpy
#import typing as tp
#import nptyping as npt


#from collections import defaultdict

#from quaternion import quaternion_mul, quaternion_inv, quaternion_vec_mul

#from sklearn.preprocessing import LabelEncoder

def load_dataset(path_to_dataset_dir : str, 
                feature_names : Optional[List[str]] = None,
                gesture_names : Optional[str] = None,
                index_file : Optional[str] = None,
                handedness: Optional[str] = None,
                ignore_handedness_both : bool = True,
                return_gesture_names : bool = False) -> Union[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray, List[str]]]:
    """
    Loads the Hand Gesture Dataset
    :param path_to_dataset_dir: str path to the main dataset folder containing *.bin files
    :param feature_names: list of features to import from the dataset. Can be any of 
                          ['HandPos','HandOrientation',
                           'GlobalHandPos','GlobalHandOrientation',
                           'HMDPos','HMDOrientation', 'Time']
    :param gesture_names: List of string with gesture naemes. Instances of the specified gestures will be imported from the dataset.
    :param index_file: Optional index file to use containing the bin filenames to use while loading the dataset
    :param handedness: [None, 'left','right'] Choose whether to load only samples of specific handedness | TODO: add support for 'both'
    :param ignore_handedness_both: If true, bin files containing recordings with both hands will be ignored while loading. Otherwise an exception will be thrown if a bin file with recordings from both hands is being found.
    :param return_gesture_names: if true, it will also return a list of gesture names for each one of the instances as the last element of the returned tuple
    :returns: tuple (pd.DataFrame, np.ndarray, List[str]) each row of the pandas DataFrame corresponds to a single gesture instance. Each column to a different feature containing a time series (pd.Seris of the measurements across time),
                np.ndarray of label integer labels corresponding to each gesture, List[str] the gesture names for each gesture instance.
     """

    data_getters = {
        'RightHandGlobalOrientation' : itemgetter('RightHandGlobalOrientation'),
        'RightHandGlobalPos' : itemgetter('RightHandGlobalPos'),
        'LeftHandGlobalOrientation' : itemgetter('LeftHandGlobalOrientation'),
        'LeftHandGlobalPos' : itemgetter('LeftHandGlobalPos'),
        'RightHandPos' : itemgetter('RightHandPos'),
        'RightHandOrientation' : itemgetter('RightHandOrientation'),
        'LeftHandPos' : itemgetter('LeftHandPos'),
        'LeftHandOrientation' : itemgetter('LeftHandOrientation'),
        'HMDPos' : itemgetter('HMDPos'),
        'HMDOrientation' : itemgetter('HMDOrientation'),
        'Time' : itemgetter('Time')
    }

    dataset_instances = [ ]
    dataset_labels = [ ]
    dataset_gesture_names = [ ]

    # if no specific feature_names were defined, import all features
    if feature_names is None:
        feature_names = list(data_getters.keys())

    if index_file is None:
        files = glob(path_to_dataset_dir+"/*.bin")
    else:
        with open(index_file,'r') as f:
            # reader = csv.reader(f)
            reader = DictReader(f)
            files = [os.path.join(path_to_dataset_dir, row['filename']) for row in reader]

    for filename in files:
        with open(filename,'rb') as f:
            sample_data = msgpack.load(f)
            sample_gesture_name = sample_data['GestureName']
            # check if HandPos/HandOrientation is defined in feature_names
            requires_pos_or_orientation = 'HandPos' in feature_names or 'HandOrientation' in feature_names
            # make sure, the bin file does not contain recordings from both controllers which would lead to ambiguity
            has_left = 'LeftHandPos' in sample_data or 'LeftHandOrientation' in sample_data
            has_right = 'RightHandPos' in sample_data or 'RightHandOrientation' in sample_data
            if requires_pos_or_orientation:
                if has_left and has_right:
                    if ignore_handedness_both:
                        continue
                    else:
                        raise Exception('WARNING: Bin file contains a recording with both hand which is not supported')

            sample_handedness = 'left' if has_left else 'right'

            gesture_feature_names = [ ]
            for feature_name in feature_names:
                if feature_name == "HandPos":
                    feature_name = 'LeftHandPos' if has_left else 'RightHandPos'
                elif feature_name == "HandOrientation":
                    feature_name = 'LeftHandOrientation' if has_left else 'RightHandOrientation'
                elif feature_name == 'GlobalHandPos':
                    feature_name = 'LeftHandGlobalPos' if has_left else 'RightHandGlobalPos'
                elif feature_name == 'GlobalHandOrientation':
                    feature_name = 'LeftHandGlobalOrientation' if has_left else 'RightHandGlobalOrientation'
                gesture_feature_names.append(feature_name)
              
            # check if we should import this gesture.
            import_gesture = gesture_names is None or sample_gesture_name in gesture_names
            if import_gesture and (handedness is None or handedness == sample_handedness):
                sample_label = sample_data['GestureID']

                # import all pre-defined feature names
                sample_features = [ ]
                for feature_name in gesture_feature_names:
                    features = np.stack(data_getters[feature_name](sample_data))
                    if features.ndim == 1:                          # hand single variable data (e.g. time)
                        features = np.expand_dims(features,1)
                    # append as pandas.Series                    
                    for idx in range(features.shape[1]):
                        sample_features.append(pd.Series(features[:,idx]))
                    

                dataset_instances.append(sample_features)
                dataset_labels.append(sample_label)
                dataset_gesture_names.append(sample_gesture_name)
                        
    ret_dataset = pd.DataFrame(dataset_instances)
    ret_labels = pd.Series(dataset_labels)
    return (ret_dataset, ret_labels) if not return_gesture_names else (ret_dataset, ret_labels, return_gesture_names)

# def load_dataset_wtime(path_to_dataset_dir : str) -> tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]]:

#     ds = []
#     for filename in glob(path_to_dataset_dir+"/*.bin"):
#         with open(filename,'rb') as f:
#             data = msgpack.load(f)
#             dp = { }
#             dp['Label'] = data['GestureID']
#             dp['GestureName'] = data['GestureName']
#             dp['RightControllerPos'] = np.stack(list(map(itemgetter('Position'),data['RightControllerData'])))
#             dp['RightControllerOrientation'] = np.stack(list(map(itemgetter('Orientation'),data['RightControllerData'])))
#             dp['LeftControllerPos'] = np.stack(list(map(itemgetter('Position'),data['LeftControllerData'])))
#             dp['LeftControllerOrientation'] = np.stack(list(map(itemgetter('Orientation'),data['LeftControllerData'])))
#             dp['HeadsetPos'] = np.stack(list(map(itemgetter('Position'),data['HeadsetData'])))
#             dp['HeadsetOrientation'] = np.stack(list(map(itemgetter('Orientation'),data['HeadsetData'])))
#             dp['Time'] = np.expand_dims(np.stack(list(map(itemgetter('Time'),data['RightControllerData']))),1)
#             ds.append(dp)
#     return ds

# def filter_dataset(ds: tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]], classes : tp.List[str]) -> tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]]:

#     fds = list(filter(lambda x: x['GestureName'] in classes, ds))
#     for v in fds:
#         v['Label'] = classes.index(v['GestureName'])        # re-label according to the filtered class names
#     return fds

# def filter_exclude_dataset(
#         ds: tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]], classes : tp.List[str], sample_count : int = 1000) -> tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]]:

#     fds = list(filter(lambda x: x['GestureName'] not in classes, ds))
#     for v in fds:
#         v['GestureName'] = 'Negative'
#         v['Label'] = 333333   # magic number

#     fds = choices(fds,k=sample_count)    
#     return fds

# def augment_negatives(ds : tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]], max_trim_percent : float = 0.5, negative_count: int = 1000):
#     aug_ds = []            
#     for i in range(negative_count):
#         # select a random gesture from the dataset
#         gest_id = randint(0,len(ds)-1)
#         ts_count = len(ds[gest_id]['Time'])
#         # produce a sequence of at most max_trim_percent percent of the sequence
#         seq_len = int(random() * max_trim_percent * ts_count + 3.5)
#         seq_start = randint(0,ts_count - seq_len)
        
#         seq_stop = min(seq_start + seq_len,ts_count-1)
#         v = ds[gest_id]
        
#         dp = { }

#         dp['GestureName'] = 'Negative'
#         dp['Label'] = 333333   # magic number
#         dp['RightControllerPos'] = v['RightControllerPos'][seq_start:seq_stop]
#         dp['RightControllerOrientation'] = v['RightControllerOrientation'][seq_start:seq_stop]
#         dp['LeftControllerPos'] = v['LeftControllerPos'][seq_start:seq_stop]
#         dp['LeftControllerOrientation'] = v['LeftControllerOrientation'][seq_start:seq_stop]
#         dp['HeadsetPos'] = v['HeadsetPos'][seq_start:seq_stop]
#         dp['HeadsetOrientation'] = v['HeadsetOrientation'][seq_start:seq_stop]
#         dp['Time'] = v['Time'][seq_start:seq_stop]
        
#         aug_ds.append(dp)

#     return aug_ds

# def augment_time_scale(ds : tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]], sigma_time : float, times: int):
    
#     aug_ds = []
#     for t in range(times):
#         for v in ds:
#             time_scale = (np.random.rand()-0.5)/0.5*sigma_time + 1.0
#             dp = { }
#             dp['GestureName'] = v['GestureName']
#             dp['Label'] = v['Label']
#             dp['RightControllerPos'] = v['RightControllerPos']
#             dp['RightControllerOrientation'] = v['RightControllerOrientation']
#             dp['LeftControllerPos'] = v['LeftControllerPos']
#             dp['LeftControllerOrientation'] = v['LeftControllerOrientation']
#             dp['HeadsetPos'] = v['HeadsetPos']
#             dp['HeadsetOrientation'] = v['HeadsetOrientation']
#             dp['Time'] = v['Time']*time_scale
#             aug_ds.append(dp)

#     return aug_ds


# def augment_noise(ds : tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),float]]], sigma_pos : float, sigma_orient  : float, times : int):
#     aug_ds = []
#     for t in range(times):
#         for _, v in ds.items():
#             dp = { }
#             dp['Label'] = v['Label']
#             dp['RightControllerPos'] = v['RightControllerPos'] + np.random.normal(scale = sigma_pos,size=v['RightControllerPos'].shape)
#             dp['RightControllerOrientation'] = v['RightControllerOrientation'] + np.random.normal(scale=sigma_orient,size=v['RightControllerOrientation'].shape)
#             dp['LeftControllerPos'] = v['LeftControllerPos'] + np.random.normal(scale = sigma_pos,size=v['LeftControllerPos'].shape)
#             dp['LeftControllerOrientation'] = v['LeftControllerOrientation'] + np.random.normal(scale=sigma_orient,size=v['LeftControllerOrientation'].shape)
#             dp['HeadsetPos'] = v['HeadsetPos'] + np.random.normal(scale = sigma_pos,size=v['HeadsetPos'].shape)
#             dp['HeadsetOrientation'] = v['HeadsetOrientation'] + np.random.normal(scale = sigma_orient, size=v['HeadsetOrientation'].shape)
#             dp['Time'] = v['Time']
#             aug_ds.append(dp)

#     return aug_ds

# def transform_pos(positions : npt.NDArray[(tp.Any,3),float], headset_pos : npt.NDArray[(tp.Any,3),float], headset_orientation : npt.NDArray[(tp.Any,4),float]) -> npt.NDArray[(tp.Any,3),float]:
#     #hpos = np.concatenate((headset_pos,np.zeros((headset_pos.shape[0],1))),axis=1)
#     #epos = np.concatenate((positions,np.zeros((positions.shape[0],1))),axis=1)

#     relativepos = quaternion_vec_mul(quaternion_inv(headset_orientation),(positions - headset_pos))
#     return relativepos

# def transform_orientation(orientations : npt.NDArray[(tp.Any,4),float], headset_orientation : npt.NDArray[4,float]) -> npt.NDArray[(tp.Any,3),float]:
#     return quaternion_mul(quaternion_inv(headset_orientation),orientations)

# # this will transform each sequence of Headset/Left/Right Controller wrt to first point in sequence

# def convert_to_panel(ds : tp.List[tp.Dict[str,npt.NDArray[(tp.Any,tp.Any),npt.Float]]], feature_names : tp.List[str]):

#     instances = []
#     labels = []
#     for sample in ds:
#         sample_feats = []
#         for fname in feature_names:
#             current_feats = [pd.Series(sample[fname][:,idx]) for idx in range(sample[fname].shape[1])]
#             sample_feats += current_feats

#         instances.append(sample_feats)
#         labels.append(sample['Label'])

#     # label encoding
#     label_enc = LabelEncoder()
#     labels = label_enc.fit_transform(labels)                # enforce labels in 0-class_count-1

#     dfFeats = pd.DataFrame(instances)
#     dfLabels = pd.Series(labels)

#     return dfFeats, dfLabels

if __name__ == "__main__":
    ds = load_dataset(
        path_to_dataset_dir=r'C:\Users\giorgospap\Desktop\EKETA\INFINITY\hand-gesture-dataset\bin\fixed-labels',
        feature_names = ['HandPos','HandOrientation','Time'],
        gesture_names = ['LeftOpenPalm','LeftRotate','LeftInfinity','LeftGrab','RightGrab'], 
        handedness = 'left'      
    )

    x=1