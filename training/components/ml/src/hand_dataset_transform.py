#### Script that pre-processes the hand gesture recognition dataset in its primitive json form
#### It converts the json files to message-pack serialized bin files
####
import msgpack
import json
import os
from glob import glob
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from itertools import compress
import csv

def first(iterable):
    return next(iter(iterable))

def parse_args():
    parser = ArgumentParser()
    
    commands_parser = parser.add_subparsers()
    
    convert_parser = commands_parser.add_parser("convert")    
    convert_parser.add_argument("--hand_dataset_source_path", type=str, required=True, help='Path to dataset folder containing had gesture json recordings')
    convert_parser.add_argument("--hand_dataset_destination_path", type=str, required=True, help = 'Path to output dataset folder that will contain the transformed .bin files (msgpack)')
    convert_parser.add_argument("--hand_dataset_filter_json_path", type=str, default = None, help = 'Optional file that contains true/false values on which json files are valid')
    convert_parser.add_argument("--default_filter_policy", choices = ['accept','reject'], default='accept', help='Default filter policy when hand_dataset_filter_json_path is provided. The policy defines what happens when the json file name in source folder is not contained in the filter json file.')
    convert_parser.set_defaults(action=convert_bin)

    fixlabels_parser = commands_parser.add_parser("fix-labels")
    fixlabels_parser.add_argument("--hand_dataset_bin_path", type=str, required=True, help='Path to the source hand gesture dataset folder containing bin files (msgpack)')
    fixlabels_parser.add_argument("--hand_dataset_output_path", type=str, default=None, help = 'Path to the output folder that will contain the same gestures as the source folder but with gesture labels modifed based on the class_label_map file')
    fixlabels_parser.add_argument("--class_label_map_file_path", type=str, default=None, help = 'Path to class_label_map_file containing metadata for the gesture classes (especially gesture id numbers for each class name)')
    fixlabels_parser.set_defaults(action=fix_labels)

    build_index_parser = commands_parser.add_parser("build-index-meta")
    build_index_parser.add_argument("--hand_dataset_bin_path", type=str, required=True, help='Path to the source hand gesture dataset folder containing bin files (msgpack)')    
    build_index_parser.add_argument("--index_meta_output_file_path", type=str, required=True, help = 'Path to the output index file that will contain label names/ids and statistics for all gestures in the dataset')    
    build_index_parser.add_argument("--detailed_index_meta_output_file_path", type=str, required=True, help = 'Path to the detailed output index file that contains meta data for each bin file in the dataset')
    build_index_parser.set_defaults(action=build_index_meta)

    args = parser.parse_args()
    return args

def write_gesture_bin(gesture_data, out_gesturef: str):    
    transformed_gesture_data = { }
    hasLeftHand = len(gesture_data['frameData']['left_hand']) > 0
    hasRightHand = len(gesture_data['frameData']['right_hand']) > 0
    for hasData, hand_name, label_prefix in [(hasLeftHand, 'left_hand', 'LeftHand'),(hasRightHand, 'right_hand', 'RightHand')]:
        if not hasData: continue
        hand_data = gesture_data['frameData'][hand_name]
        #for time_data in hand_data:
        #    pos_vec = np.array([[pos_data['x'], pos_data['y'], pos_data['z']] for pos_data in time_data['pos']]).reshape(-1)
        pos_vec = np.stack(list(map(lambda time_data: np.array([[pos_data['x'], pos_data['y'], pos_data['z']] for pos_data in time_data['pos']]).reshape(-1), hand_data)))
        orient_vec = np.stack(list(map(lambda time_data: np.array([[orient_data['x'], orient_data['y'], orient_data['z'], orient_data['w'] ] for orient_data in time_data['rotation']]).reshape(-1), hand_data)))

        transformed_gesture_data[label_prefix+"Pos"] = pos_vec.tolist()
        transformed_gesture_data[label_prefix+"Orientation"] = orient_vec.tolist()

    for hasData, global_name, label_prefix in [(hasLeftHand, 'left_hand_global', 'LeftHandGlobal'),
                                                (hasRightHand, 'right_hand_global', 'RightHandGlobal'),
                                                (True, 'hmd', 'HMD') 
                                               ]:
        if not hasData: continue
        global_data       = gesture_data['frameData'][global_name]
        global_pos_vec    = np.array([[pos_data['x'], pos_data['y'], pos_data['z']] for pos_data in global_data['pos']])
        global_orient_vec = np.array([[orient_data['x'], orient_data['y'], orient_data['z'], orient_data['w']] for orient_data in global_data['rotation']])
        
        transformed_gesture_data[label_prefix+"Pos"] = global_pos_vec.tolist()
        transformed_gesture_data[label_prefix+"Orientation"] = global_orient_vec.tolist()
    
    transformed_gesture_data['Time'] = gesture_data['frameData']['timestamps']
    transformed_gesture_data['GestureName'] = gesture_data['class_name']
    transformed_gesture_data['GestureID'] = gesture_data['class_ID']
    
    with open(out_gesturef, 'wb') as f:
        msgpack.dump(transformed_gesture_data, f)

def convert_bin(args):
    os.makedirs(args.hand_dataset_destination_path,exist_ok=True)
    input_gesture_filenames = glob(os.path.join(args.hand_dataset_source_path,"*.json"))
    out_gesture_filenames = [os.path.join(args.hand_dataset_destination_path, f'gesture-{gesture_id}.bin') for gesture_id in range(len(input_gesture_filenames))]

    if args.hand_dataset_filter_json_path is not None:
        with open(args.hand_dataset_filter_json_path, 'r') as f:
            flt : dict = json.load(f)

            filtered = list(map(lambda filename: flt.get(os.path.basename(filename),args.default_filter_policy == 'accept'), input_gesture_filenames))
            input_gesture_filenames = list(compress(input_gesture_filenames, filtered))
            out_gesture_filenames = list(compress(out_gesture_filenames, filtered))

    for in_gesturef, out_gesturef in tqdm(zip(input_gesture_filenames, out_gesture_filenames), total = len(input_gesture_filenames)):
        with open(in_gesturef, 'r') as f:
            gesture_data = json.load(f)
            write_gesture_bin(gesture_data, out_gesturef)

def fix_labels(args):
    if args.class_label_map_file_path is None:
        raise Exception("You need to provide class_label_map_file_path")

    os.makedirs(args.hand_dataset_output_path, exist_ok=True)
    in_files = glob(os.path.join(args.hand_dataset_bin_path,"*.bin"))
    out_files = list(map(lambda x: os.path.join(args.hand_dataset_output_path, os.path.basename(x)), in_files))

    label_map = { }
    with open(args.class_label_map_file_path, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            label_map[row['gesture_name']] = int(row['gesture_id'])

    for in_fname, out_fname in tqdm(zip(in_files, out_files), total = len(in_files)):
        with open(in_fname,'rb') as f:
            gesture_data = msgpack.load(f)
        new_gesture_id = label_map[gesture_data['GestureName']]
        gesture_data['GestureID'] = new_gesture_id
        with open(out_fname,'wb') as f:
            msgpack.dump(gesture_data, f)

def build_index_meta(args):
    os.makedirs(os.path.dirname(args.index_meta_output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.detailed_index_meta_output_file_path), exist_ok=True)

    in_files = glob(os.path.join(args.hand_dataset_bin_path,"*.bin"))
    gesture_names = [ ]
    gesture_ids = [ ]
    handedness = [ ]
    index_meta = [ ]    

    for inf in tqdm(in_files):
        
        with open(inf, 'rb') as f:
            gesture_data = msgpack.load(f)
            gesture_names.append(gesture_data['GestureName'])
            gesture_ids.append(gesture_data['GestureID'])
            is_right = 'RightHandPos' in gesture_data
            is_left = 'LeftHandPos' in gesture_data
            if is_left and is_right:
                handedness.append('both')
            elif is_left:
                handedness.append('left')
            elif is_right:
                handedness.append('right')

    # sanity check
    for gest_id in set(gesture_ids):
        filtered = (np.array(gesture_ids) == gest_id).tolist()
        assert len(set(compress(gesture_names,filtered))) == 1
    assert len(set(gesture_names)) == len(set(gesture_ids))

    # statistics
    # for gesture_name in sorted(set(gesture_names)):
    for gesture_id in sorted(set(gesture_ids)):
        filtered = list(map(lambda gest_id: gest_id == gesture_id, gesture_ids))
        gesture_name = first(compress(gesture_names,filtered))
        total_count = np.array(filtered).sum()
        
        assert all(map(lambda x: x == 'left', compress(handedness, filtered))) or all(map(lambda x: x == 'right', compress(handedness, filtered))) or \
                                    all(map(lambda x: x=='both', compress(handedness,filtered)))
        
        meta = {
            'gesture_id': gesture_id,
            'gesture_name': gesture_name,
            'total_count': total_count,
            'handedness' : first(compress(handedness, filtered))
        }
        index_meta.append(meta)
    
    detailed_index_meta = [(os.path.basename(in_file), gesture_id, gesture_name, hand) for in_file, gesture_id, gesture_name, hand in 
                                zip(in_files, gesture_ids, gesture_names, handedness)                           
                          ]

    # write statistics
    with open(args.index_meta_output_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('gesture_id','gesture_name','total_count','handedness'))
        writer.writeheader()
        writer.writerows(index_meta)

    # write detailed index
    with open(args.detailed_index_meta_output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','gesture_id','gesture_name','handedness'])
        writer.writerows(detailed_index_meta)

def main():
    args = parse_args()
    args.action(args)
    
if __name__ == "__main__":
    main()