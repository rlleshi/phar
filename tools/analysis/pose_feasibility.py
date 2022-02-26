import os
import os.path as osp
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import seaborn as sns
from rich.console import Console
from tqdm import tqdm

sys.path.append('./tools')  # noqa: E501
import utils as utils  # noqa isort:skip

CONSOLE = Console()
POSE_EXTR_PATH = 'tools/data/skeleton/pose_extraction.py'


def parse_args():
    parser = ArgumentParser(prog='check whether for a particular action class '
                            'it makes sense to do skeleton-based HAR.'
                            'Also generates the .pkl pose dicts.')
    parser.add_argument('label', help='class/label to examine')
    parser.add_argument('--src-dir',
                        default='mmaction2/data/phar',
                        help='directory of dataset')
    parser.add_argument('--out_dir', default='mmaction2/data/phar/pose')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations.txt',
                        help='annotation file')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='the splits where clips are found')
    parser.add_argument('--pose-score-thr',
                        type=float,
                        default=0.6,
                        help='pose estimation score threshold')
    parser.add_argument('--device', default='cuda:0', help='device')
    args = parser.parse_args()
    return args


def get_pose(video, out_dir, args):
    # based on the scripts output
    needle = '% of poses are incorect'
    # TODO: loading the model every time we run the command slows down things
    # TODO: if the model was only loaded once, things would be much faster
    subargs = [
        'python', POSE_EXTR_PATH, video, args.ann, '--out-dir', out_dir,
        '--device', args.device, '--pose-score-thr',
        str(args.pose_score_thr)
    ]
    result = subprocess.run(subargs, capture_output=True)
    result = utils.prettify(result.stdout).split(needle)[0][-6:-1]
    try:
        result = float(result)
    except ValueError:
        result = -1
    return result


def main():
    args = parse_args()
    # how many clips have X% of poses wrong
    results = {k: 0 for k in range(0, 101, 10)}

    for split in args.splits:
        out_dir = osp.join(args.out_dir, split, args.label)
        in_dir = osp.join(args.src_dir, split, args.label)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for clip in tqdm(os.listdir(in_dir)):
            result = get_pose(osp.join(in_dir, clip), out_dir, args)
            if result == -1:
                continue

            for k in results.keys():
                if result > k:
                    results[k] += 1

    # plot
    df = pd.DataFrame({
        '%': list(results.keys()),
        'Value': list(results.values())
    })
    sns.set(rc={'figure.figsize': (15, 13)})
    fig = sns.barplot(x='%', y='Value', data=df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30)
    fig.axes.set_title(f'Incorrect Poses ({args.pose_score_thr}-conf-thr)',
                       fontsize=40)
    fig.set_xlabel('%', fontsize=30)
    fig.set_ylabel('Values', fontsize=20)
    output = fig.get_figure()

    out = osp.join(args.out_dir, f'incorrect_poses_{args.label}.jpg')
    output.savefig(out)
    CONSOLE.print(f'Saved @{out}')


if __name__ == '__main__':
    main()
