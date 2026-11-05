from dataset.common import generate_train_test_split_index_file
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path_to_dataset_dir", type=str,required=True, help='Path to dataset dir containing bin files')
    parser.add_argument("--out_path_to_train_split_txt", type=str,required=True)
    parser.add_argument("--out_path_to_test_split_txt", type=str,required=True)
    parser.add_argument("--gesture_names", nargs = "+", type=str, default=None)
    parser.add_argument("--train_size", type=float, default=0.7, help='Percent (0.0-1.0) of the training set')
    parser.add_argument("--random_state",type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    generate_train_test_split_index_file(
        path_to_dataset_dir = args.path_to_dataset_dir,
        out_path_to_train_split_txt = args.out_path_to_train_split_txt,
        out_path_to_test_split_txt = args.out_path_to_test_split_txt,
        gesture_names = args.gesture_names,
        train_size = args.train_size,
        random_state = args.random_state
    )
if __name__ == "__main__":
    main()
