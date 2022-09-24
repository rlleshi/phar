import abc
import os
import os.path as osp
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import seaborn as sns
from rich.console import Console
from tqdm import tqdm

sys.path.append('./src')  # noqa: E501
import data.pose_extraction as pose_extraction  # noqa isort:skip

try:
    from mmdet.apis import init_detector
    from mmpose.apis import init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for pose extraction.')  # noqa: E501

CONSOLE = Console()
POSE_EXTR_PATH = 'src/data/skeleton/pose_extraction.py'
PROGRESS_FILE = 'pose_feasibility_progress.txt'


def parse_args():
    parser = ArgumentParser(prog='check the pose feasibility for a class'
                            'Also generates the .pkl pose dicts.')
    parser.add_argument('label', help='class/label to examine')
    parser.add_argument('--src-dir',
                        default='mmaction2/data/phar',
                        help='directory of dataset')
    parser.add_argument('--out-dir', default='mmaction2/data/phar/pose')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/pose.txt',
                        help='annotation file')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='the splits where clips are found')
    parser.add_argument('--pose-score-thr',
                        type=float,
                        default=0.2,
                        help='pose estimation score threshold')
    parser.add_argument('--resume',
                        action='store_true',
                        help='ggf. resume analysis from previous run')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument(
        '--det-config',
        default=('mmdetection/configs/faster_rcnn/'
                 'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'),
        help='human detector config')
    parser.add_argument(
        '--det-checkpoint',
        default='checkpoints/detector/faster_rcnn_r50_fpn_1x_coco-person.pth',
        help='human detector checkpoint')
    parser.add_argument(
        '--pose-config',
        default=('mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
                 'coco/hrnet_w32_coco_256x192.py'),
        help='pose estimation config')
    parser.add_argument('--pose-checkpoint',
                        default='checkpoints/pose/hrnet_w32_coco_256x192.pth',
                        help='pose estimation checkpoint')
    args = parser.parse_args()
    return args


def get_pose(args, d_model, p_model):
    """Perform pose estimation given a video.

    Args:
        args (dict): parsed args
        d_model: detection model
        p_model: pose model

    Returns:
        int: correct poses rate
    """
    return pose_extraction.main(args, d_model, p_model)


def main():
    args = parse_args()
    # percentiles of clips having correct poses:
    # for a certain percentile it means: {n_videos_in_percentile / total_vids}
    # have {percentile %} of their poses with a confidence higher than
    # {args.pose_score_thr}
    results = {k: 0 for k in range(0, 101, 10)}
    det_model = init_detector(args.det_config, args.det_checkpoint,
                              args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)

    sub_args = abc.ABC()
    sub_args = abc.abstractproperty()
    sub_args.device = args.device
    sub_args.det_score_thr = 0.5
    sub_args.pose_score_thr = args.pose_score_thr
    sub_args.ann = args.ann
    sub_args.correct_rate = 0.2
    sub_args.filter_pose = False

    resume_list = []
    if args.resume:
        if osp.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as resume_from:
                resume_list = resume_from.readlines()[0].split(',')
        else:
            CONSOLE.print(
                f'Resume option selected but {PROGRESS_FILE} not'
                ' found.',
                style='yellow')

    for split in args.splits:
        out_dir = osp.join(args.out_dir, split, args.label)
        in_dir = osp.join(args.src_dir, split, args.label)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        sub_args.out_dir = out_dir

        for clip in tqdm(os.listdir(in_dir)):
            if clip in resume_list:
                CONSOLE.print(f'Already processed. Skipping {clip}.',
                              style='green')
                continue

            sub_args.video = osp.join(in_dir, clip)
            result = get_pose(sub_args, det_model, pose_model)
            if result is None:
                CONSOLE.print(f'{clip} already exists. Skipping.',
                              style='green')
                continue

            result *= 100
            for k in results.keys():
                if result > k:
                    results[k] += 1
            with open(PROGRESS_FILE, 'a+') as out:
                out.write(f'{clip},')

    # plot
    df = pd.DataFrame({
        '%': list(results.keys()),
        'Value': list(results.values())
    })
    sns.set(rc={'figure.figsize': (15, 13)})
    fig = sns.barplot(x='%', y='Value', data=df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30)
    fig.axes.set_title(f'Correct Poses ({args.pose_score_thr}-conf-thr)',
                       fontsize=40)
    fig.set_xlabel('%', fontsize=30)
    fig.set_ylabel('Values', fontsize=20)
    output = fig.get_figure()

    out = osp.join(args.out_dir, f'correct_poses_rate_{args.label}.jpg')
    output.savefig(out)
    CONSOLE.print(f'Saved @{out}')


if __name__ == '__main__':
    main()
