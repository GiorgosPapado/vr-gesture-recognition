import msgpack
import json
import os
from glob import glob
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from itertools import compress
from operator import itemgetter
import csv

def first(iterable):
    return next(iter(iterable))

def parse_args():
    parser = ArgumentParser()
    
    commands_parser = parser.add_subparsers()
    
    remove_unused_controller_parser = commands_parser.add_parser("remove-unused-controller")    
    remove_unused_controller_parser.add_argument("--controller_dataset_source_path", type=str, required=True, help='Path to dataset folder containing controller gesture bin recordings')
    remove_unused_controller_parser.add_argument("--controller_dataset_destination_path", type=str, required=True, help = 'Path to output dataset folder that will contain the same bin files but data for the unused controller removed')    
    remove_unused_controller_parser.set_defaults(action=remove_unused_controller)

    build_index_parser = commands_parser.add_parser("build-index-meta")
    build_index_parser.add_argument("--controller_dataset_source_path", type=str, required=True, help='Path to the source controller gesture dataset folder containing bin files (msgpack)')
    build_index_parser.add_argument("--index_meta_output_file_path", type=str, required=True, help = 'Path to the output index file that will contain label names/ids and statistics for all gestures in the dataset')    
    build_index_parser.add_argument("--detailed_index_meta_output_file_path", type=str, required=True, help = 'Path th the detailed output index file that contains meta data for each bin file in the dataset')
    build_index_parser.set_defaults(action=build_index_meta)
    args = parser.parse_args()
    return args


def compute_trajectory_length(position_data : np.ndarray) -> float:
    return np.linalg.norm(np.diff(position_data,axis=0),axis=1).sum()

def remove_unused_controller(args):    
    os.makedirs(args.controller_dataset_destination_path, exist_ok=True)
    in_files = glob(os.path.join(args.controller_dataset_source_path,"*.bin"))
    out_files = list(map(lambda fname: os.path.join(args.controller_dataset_destination_path, os.path.basename(fname)), in_files))

    for inf, outf in tqdm(zip(in_files, out_files), total = len(in_files)):
        with open(inf, 'rb') as f:
            gesture_data : dict = msgpack.load(f)
            left_controller_trajectory_length = compute_trajectory_length(np.array(list(map(itemgetter('Position'), gesture_data['LeftControllerData']))))
            right_controller_trajectory_length = compute_trajectory_length(np.array(list(map(itemgetter('Position'), gesture_data['RightControllerData']))))
            if left_controller_trajectory_length < right_controller_trajectory_length:
                gesture_data.pop('LeftControllerData')
            else:
                gesture_data.pop('RightControllerData')
            
            with open(outf, 'wb') as fw:
                msgpack.dump(gesture_data, fw)

def build_index_meta(args):
    os.makedirs(os.path.dirname(args.index_meta_output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.detailed_index_meta_output_file_path), exist_ok=True)

    in_files = glob(os.path.join(args.controller_dataset_source_path,"*.bin"))
    gesture_names = [ ]
    gesture_ids = [ ]
    is_right = [ ]
    index_meta = [ ]    

    for inf in tqdm(in_files):
        
        with open(inf, 'rb') as f:
            gesture_data = msgpack.load(f)
            gesture_names.append(gesture_data['GestureName'])
            gesture_ids.append(gesture_data['GestureID'])
            is_right.append('RightControllerData' in gesture_data)

    # sanity check
    for gest_id in set(gesture_ids):
        filtered = (np.array(gesture_ids) == gest_id).tolist()
        assert len(set(compress(gesture_names,filtered))) == 1
    assert len(set(gesture_names)) == len(set(gesture_ids))

    # statistics
    #for gesture_name in sorted(set(gesture_names)):
    for gesture_id in sorted(set(gesture_ids)):
        filtered = list(map(lambda gest_id: gest_id == gesture_id, gesture_ids))
        gesture_name = first(compress(gesture_names,filtered))
        total_count = np.array(filtered).sum()
        right_count = np.array(list(compress(is_right, filtered))).sum()
        left_count = np.array(list(compress(map(lambda x: not x, is_right), filtered))).sum()
        meta = {
            'gesture_id': gesture_id,
            'gesture_name': gesture_name,
            'total_count': total_count,
            'left_count': left_count,
            'right_count': right_count
        }
        index_meta.append(meta)
    
    # detailed index
    is_right_to_str = {
        True: 'right',
        False: 'left'
    }
    detailed_index_meta = [(os.path.basename(in_file), gesture_id, gesture_name, is_right_to_str[isright]) for in_file, gesture_id, gesture_name, isright in 
                                zip(in_files, gesture_ids, gesture_names, is_right)
                           ]

    # write statistics
    with open(args.index_meta_output_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('gesture_id','gesture_name','total_count','left_count','right_count'))
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