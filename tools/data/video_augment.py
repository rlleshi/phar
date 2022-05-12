import argparse
import glob
import os.path as osp
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from tqdm import tqdm
from vidaug import augmentors as va

sys.path.append('./tools')  # noqa
import utils as utils  # noqa isort:skip

CONSOLE = Console()
AUGS = [
    va.InvertColor(),
    va.Add(value=125),
    va.Add(value=-125),
    va.Pepper(ratio=50),
    va.Salt(ratio=100),
    va.GaussianBlur(sigma=2),
    va.GaussianBlur(sigma=3.5),
    va.ElasticTransformation(alpha=3.5, sigma=0.5),
    va.PiecewiseAffineTransform(displacement=4,
                                displacement_kernel=2,
                                displacement_magnification=3),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Augmenting train set script')
    parser.add_argument('--src-dir',
                        default='mmaction2/data/phar/train',
                        help='source video directory')
    parser.add_argument('--out-dir',
                        default='mmaction2/data/phar/train_aug/',
                        help='augmented video directory')
    parser.add_argument('--rate',
                        type=float,
                        default=0.3,
                        help='replacement rate for videos')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations.txt',
                        help='annotation file')
    args = parser.parse_args()
    return args


def augment_video(clip: str, out_dir: str):
    """Augments a video.

    Args:
        clip (str): path to video
        out_dir (str): path to out dir
    """
    video = cv2.VideoCapture(clip)
    out = osp.join(out_dir, osp.basename(clip))
    video_writer = cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*'MP4V'), video.get(cv2.CAP_PROP_FPS),
        (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
         round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frames = []
    while cv2.waitKey(1) < 0:
        success, frame = video.read()
        if not success:
            video.release()
            break
        frames.append(frame)

    aug = random.choice(AUGS)
    frames = aug(np.array(frames))
    for frame in frames:
        video_writer.write(frame)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for label in tqdm(utils.annotations_list(args.ann)):
        out_dir_label = osp.join(args.out_dir, label)
        Path(out_dir_label).mkdir(parents=True, exist_ok=True)
        clips = glob.glob(osp.join(args.src_dir, label, '*'))
        random_clips = random.sample(clips, round(len(clips) * args.rate))
        CONSOLE.print(f'Augmenting {len(random_clips)} clips for {label}...',
                      style='bold green')

        for clip in clips:
            if clip not in random_clips:
                shutil.copy(clip, out_dir_label)
            else:
                augment_video(clip, out_dir_label)


if __name__ == '__main__':
    main()
