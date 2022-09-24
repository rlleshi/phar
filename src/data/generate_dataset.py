import glob
import os.path as osp
import random
import string
import sys
from argparse import ArgumentParser
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path

import moviepy.editor as mpy
import numpy as np
import pandas as pd
from rich.console import Console
from tqdm.contrib.concurrent import process_map

sys.path.append('./src')  # noqa: E501
import utils as utils  # noqa isort:skip

CONSOLE = Console()

VIDEO_EXTS = ['mp4']
ANN_EXT = '.csv'
ANN_TO_INDEX = dict()


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def generate_structure(out_dir: str, annotations: str):
    """Generate the videos dataset structure.

    Args:
        out_dir (str): directory to generate structure for
        annotations (str): path to annotation file that has classes
    """
    classes = utils.annotations_list(annotations)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for c in classes:
            Path(osp.join(out_dir, split, c)).mkdir(parents=True,
                                                    exist_ok=True)
        open(osp.join(out_dir, f'{split}.txt'), 'w').close()


def get_video_annotation(id: int, anns: list) -> str:
    """Gets the annotation for a video based on its id. The assumption here is
    that both the video and its corresponding annotation have been named with a
    number.

    Args:
        id (int): video id
        anns (list): list of annotation paths

    Returns:
        ann (str): path to annotations
    """
    return (ann for ann in anns if ann.split(osp.sep)[-1][:-4] == id)


def parse_args():
    parser = ArgumentParser(prog='generate video dataset.'
                            'Videos have the same name as annotations')
    parser.add_argument('--src_dir',
                        default='dataset/',
                        help='source video directory')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/all.txt',
                        help='annotation file')
    parser.add_argument('--out-dir',
                        default='mmaction2/data/phar',
                        help='out video directory')
    parser.add_argument('--split',
                        type=float,
                        nargs='+',
                        default=[0.8, 0.2, 0],
                        help='train/val/test split')
    parser.add_argument('--clip-len',
                        type=int,
                        default=10,
                        help='length of each clip')
    parser.add_argument('--num-processes',
                        type=int,
                        default=(cpu_count() - 2 or 1),
                        help='number of processes used')
    parser.add_argument('--level',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='directory level to find videos')
    args = parser.parse_args()
    return args


def write_annotation(path: str):
    """Write the corresponding annotation to the annotation file. The
    annotation consists of the path to the video + label number.

      `mmaction2/data/temp/train/the_snake/DOZ9WC51.mp4 9`

    Args:
        path (str): path to the clip
    """
    path_to_ann_f, label = osp.split(osp.dirname(path))
    with open(f'{path_to_ann_f}.txt', 'a') as out:
        out.write(f'{path} {ANN_TO_INDEX[label]}')
        out.write('\n')


def get_actions_with_timestamps(path: str) -> list:
    """Given the path to a csv file, get its timestamps.

    This function is specific to the temporal csv annotations
    produced by the VIA annotator:

    `Show/Hide attribute editor` -> Add `Activity`
        Name: "Activity";
        Anchor: "Temporal Segment in Video or Audio";
        Description: "Activity"

    Args:
        path (str): path to annotation

    Returns:
        list: list of timestamps
    """
    results = []
    df = pd.read_csv(path)
    for i in range(1, len(df)):
        temp = str(df.iloc[i].value_counts()).split(' ')
        results.append({
            'action':
            temp[0].split(':"')[1].strip('}"'),
            'video':
            ''.join(list(filter(lambda x: x not in '["],', temp[6]))),
            'start':
            float(temp[7][:-1]),
            'end':
            float(temp[8][:-2])
        })
    return results


def extract_clips(items):
    """Extract clips of length `args.clip_len` given a video and its
    annotations."""
    video_f, anns, args = items
    ann = get_video_annotation(id=video_f.split(osp.sep)[-1][:-4], anns=anns)
    ann = next(ann, None)
    if ann is None:
        CONSOLE.print(f'Video {video_f} has no annotations.', style='yellow')
        return

    clip_len = args.clip_len
    min_remainder = clip_len / 2  # ggf. overlap
    np.random.seed()
    split = np.random.choice(['train', 'val', 'test'], p=args.split)
    video = mpy.VideoFileClip(video_f)

    for action in get_actions_with_timestamps(ann):
        duration = action['end'] - action['start']
        if np.isnan(duration):
            # faulty annotation
            continue
        if duration < clip_len:
            continue

        label = action['action'].replace('-', '_')

        if ANN_TO_INDEX.get(label, None) is None:
            # skip if label not found
            continue

        n_clips = int(duration / clip_len)
        remainder = duration % clip_len

        for i in range(n_clips):
            start = action['start'] + i * clip_len
            end = start + clip_len
            subclip = video.subclip(start, end)
            out_f = f'{osp.join(args.out_dir, split, label, gen_id())}.mp4'

            try:
                subclip.write_videofile(out_f, logger=None)
                write_annotation(out_f)
            except OSError:
                CONSOLE.print(f'{video_f} has bad annotations', style='red')
                continue

        if remainder >= min_remainder:
            # small overlap will exist, but we savor`min_remainder` footage
            out_f = f'{osp.join(args.out_dir, split, label, gen_id())}.mp4'
            subclip = video.subclip(action['end'] - clip_len, action['end'])
            try:
                subclip.write_videofile(out_f, logger=None)
                write_annotation(out_f)
            except OSError:
                CONSOLE.print(f'{video_f} has bad annotations', style='red')
                pass


def main():
    args = parse_args()
    assert sum(args.split) == 1, 'train/val/test split must equal to 1'
    assert osp.exists(args.ann), 'provide label map file'
    generate_structure(args.out_dir, args.ann)
    global ANN_TO_INDEX
    ANN_TO_INDEX = utils.annotations_dic(args.ann)

    if args.level == 1:
        items = glob.glob(osp.join(args.src_dir, '*'))
    elif args.level == 2:
        items = glob.glob(osp.join(args.src_dir, '*', '*'))

    videos = [
        item for item in items if any(
            item.endswith(ext) for ext in VIDEO_EXTS)
    ]
    annotations = [item for item in items if item.endswith(ANN_EXT)]
    np.random.shuffle(videos)

    process_map(extract_clips,
                zip(videos, repeat(annotations), repeat(args)),
                max_workers=args.num_processes,
                total=len(videos))


if __name__ == '__main__':
    main()
