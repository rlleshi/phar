import os
import os.path as osp
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import numpy as np
from tqdm import tqdm

# (1) Build the RawFrames dataset from an existing VideosDataset by using the
#     mmaction2/tools/data/build_rawframes.py script
#     (for the testing set of 600MB it took 51GB)
# (2) Use this script to generate the file list
# (3) Use the generated dataset in a config file & fine-tune a model


def train_test_parser(val):
    """Must be between range 0 & 1."""
    try:
        val = float(val)
    except ValueError:
        raise ArgumentTypeError(f'{val} must be float')
    if val < 0.0 or val > 1.0:
        raise ArgumentTypeError('Test split out of bounds [0,1]')
    return val


def get_num_rgb(path):
    return max([
        int(osp.splitext(frame)[0][4:])
        for frame in os.listdir(path) if frame[:4] == 'img_'
    ],
               default=0)


def parse_args():
    parser = ArgumentParser(prog='generate file list for rawframes dataset')
    parser.add_argument('ann', type=str, help='annotation file')
    parser.add_argument('src_dir', type=str, help='root dir for rawframes')
    parser.add_argument(
        '--val-split',
        type=train_test_parser,
        default='0.18',
        required=False,
        help='train/validation ratio. Give for validation')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='flag to generate the testing set')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not osp.exists(args.src_dir):
        Path(args.src_dir).mkdir(parents=True, exist_ok=True)

    if args.test_set:
        open(osp.join(args.src_dir, 'rawframes_test.txt'), 'w').close()
    else:
        open(osp.join(args.src_dir, 'rawframes_train.txt'), 'w').close()
        open(osp.join(args.src_dir, 'rawframes_val.txt'), 'w').close()
    label_to_number = {}
    with open(args.ann) as ann:
        for line in ann:
            (val, key) = line.split(' ')
            label_to_number[key.strip().replace('-', '_')] = int(val)

    for cls in tqdm(os.listdir(args.src_dir)):
        cls_path = osp.join(args.src_dir, cls)
        if osp.isfile(cls_path):
            continue

        for clip in os.listdir(cls_path):
            clip_path = osp.join(cls_path, clip)
            if args.test_set:
                split = 'rawframes_test.txt'
            else:
                split = 'rawframes_train.txt' if np.random.choice(
                    [0, 1], p=[1-args.val_split, args.val_split]) == 0 \
                    else 'rawframes_val.txt'

            num_frames = get_num_rgb(clip_path)
            with open(osp.join(args.src_dir, split), 'a') as f_list:
                f_list.write((f'{args.src_dir}/{cls}/{clip} '
                              f'{num_frames} {label_to_number[cls]}\n'))


if __name__ == '__main__':
    main()
