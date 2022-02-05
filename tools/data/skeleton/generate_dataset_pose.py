import logging
import os
import os.path as osp
import random
import subprocess
import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append('human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skip

CLIPS_PATH = 'clips'
RESULT_PATH = 'results'


def generate_structure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        Path(osp.join(path, CLIPS_PATH, split)).mkdir(
            parents=True, exist_ok=True)
    Path(osp.join(path, RESULT_PATH)).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = ArgumentParser(prog='generate pose data for skeleton-based-har '
                            'based on a VideoDataset directory.')
    parser.add_argument('src_dir', type=str, help='VideoDataset directory')
    parser.add_argument(
        'split_set',
        nargs='+',
        choices=['train', 'val', 'test'],
        help='type of sets to generate the pose dataset for')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/skeleton/bast_base/',
        help='resulting dataset dir')
    parser.add_argument(
        '--ann',
        type=str,
        default=('human-action-recognition/har/annotations/BAST/base/'
                 'tanz_annotations.txt'),
        help='annotations')
    parser.add_argument(
        '--devices',
        nargs='+',
        choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        help='gpu to use; can parallelize for each split-set')
    args = parser.parse_args()
    return args


def merge_results(args):
    for split in args.split_set:
        in_dir = osp.join(args.out_dir, CLIPS_PATH, split)
        out_dir = osp.join(args.out_dir, RESULT_PATH)
        helpers.merge_pose_data(in_dir, out_dir, split)


def get_pose(video, args, split, gpu):
    script_path = ('human-action-recognition/har/tools/data/skeleton/'
                   'pose_extraction.py')
    if split == '':
        split = 'test'
    else:
        split = split.split('_')[1]

    out_dir = osp.join(args.out_dir, CLIPS_PATH, split)
    subargs = [
        'python', script_path, video, args.ann, '--out-dir', out_dir,
        '--device', gpu
    ]
    try:
        logging.info(subprocess.run(subargs))
    except subprocess.CalledProcessError as e:
        logging.exception(f'Error while generating pose data for {video}: {e}')


def extract_pose(pose_items):
    split_label, gpu, args = pose_items
    split, labels = split_label
    label_path = osp.join(args.src_dir, split)

    for label in labels:
        print(f'Extracting pose for {split} - {label}')
        clip_path = osp.join(label_path, label)

        for clip in tqdm(os.listdir(clip_path)):
            get_pose(osp.join(clip_path, clip), args, split, gpu)


def main():
    logging.basicConfig(filename='skeleton_dataset.log', level=logging.DEBUG)
    args = parse_args()
    generate_structure(args.out_dir)
    n_gpus = len(args.devices)
    pool = Pool(n_gpus)

    split_labels = []
    for split in args.split_set:
        # based on the current structure of the `data-transfer` volume
        if split == 'test':
            split = ''
        else:
            split = 'videos_' + split

        labels = os.listdir(osp.join(args.src_dir, split))
        random.shuffle(labels)
        # split_labels = [(train, [walk, ..., stamp]), ...
        #   (val, [contract_expand, ..., fall])]
        n_splits = int(n_gpus / len(args.split_set))
        split_labels += [(split, label_split)
                         for label_split in np.array_split(labels, n_splits)]

    if len(args.devices) > 1:
        pool.map(extract_pose, zip(split_labels, args.devices, repeat(args)))
    else:
        print('Running on a single GPU')
        dd = defaultdict(list)
        # merge the splits
        for key, value in split_labels:
            if len(dd[key]) == 0:
                dd[key] = value
            else:
                for v in value:
                    dd[key].append(v)
        split_labels = list(dd.items())
        for split_label in split_labels:
            extract_pose((split_label, args.devices[0], args))

    merge_results(args)


if __name__ == '__main__':
    main()
