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

sys.path.append('./tools')  # noqa: E501
import data.skeleton.pose_extraction as pose_extraction  # noqa isort:skip

try:
    from mmdet.apis import init_detector
    from mmpose.apis import init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for pose extraction.')  # noqa: E501

CONSOLE = Console()
POSE_EXTR_PATH = 'tools/data/skeleton/pose_extraction.py'


def parse_args():
    parser = ArgumentParser(prog='check the pose feasibility for a class'
                            'Also generates the .pkl pose dicts.')
    parser.add_argument('label', help='class/label to examine')
    parser.add_argument('--src-dir',
                        default='mmaction2/data/phar',
                        help='directory of dataset')
    parser.add_argument('--out_dir', default='mmaction2/data/phar/pose')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations_pose.txt',
                        help='annotation file')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='the splits where clips are found')
    # TODO: 1) Consider lowering the threshold (need to test)
    # TODO: 2) Consider not making the pose points with low confidence zero
    parser.add_argument('--pose-score-thr',
                        type=float,
                        default=0.6,
                        help='pose estimation score threshold')
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
        video (str): path to video
        out_dir (str): out dir
        args (ArgumentParser): script args
        d_model: detection model
        p_model: pose model

    Returns:
        int: incorrect poses rate
    """
    return pose_extraction.main(args, d_model, p_model)


def main():
    args = parse_args()
    # how many clips have X% of poses wrong
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
    sub_args.correct_rate = 0.5

    for split in args.splits:
        out_dir = osp.join(args.out_dir, split, args.label)
        in_dir = osp.join(args.src_dir, split, args.label)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        sub_args.out_dir = out_dir

        for clip in tqdm(os.listdir(in_dir)):
            sub_args.video = osp.join(in_dir, clip)
            result = 100 * get_pose(sub_args, det_model, pose_model)

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
