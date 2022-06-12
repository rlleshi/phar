import glob
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from rich.console import Console
from tqdm import tqdm

sys.path.append('./tools')  # noqa: E501
import utils as utils  # noqa isort:skip

CONSOLE = Console()

# Methods overview:
#
# 1. Pose Stuff
#   1.1 merge_pose(): merges poses in a single list of dicts
#   1.2 filter_pose(): filter pose estimation based on threshold
#   1.3 read_pickel(): examines the pose information
#   1.4 convert_pose_label(): e.g. from (general) annotations.txt to pose_ann
#   1.5 visualize_heatmaps(): visualize groups of poses' heatmaps
#   1.6 extract_labels_pose(): extracts the label maps from poses
#
# 2. Video Manipulation
#   2.1 merge_videos(): with MoviePy
#   2.2 resize(): resize videos using MoviePy
#   2.3 extract_subclip(): extract a subclip from a video
#   2.4 extract_frames_from_video(): extract frame from a video
#   2.5 trim_dataset(): shuffles and removes part of dataset
#
# 3. Miscellaneous
#   3.1 gen_id(): generates a 'random' ID
#   3.2 resize_img(): resizes an image using MoviePy
#   3.3 download_youtube(): downloads youtube video
#   3.4 extract_timestamps(): annotation timestamp extraction based on VIA
#   3.5 merge_images_with_font(): merge images and add fonts to them
#
# 4. Dataset Stuff
#   4.1 gen_single_ann_file(): generates annotation file for a single class
#   4.2 augment_video(): augments a single video
#
# 5. Obscure
#   5.1 merge_train_test()


def extract_labels_pose(path: str):
    import pickle
    with open(path, 'rb') as f:
        annotations = pickle.load(f)
    out_f = osp.splitext(path)[0] + '.txt'
    with open(out_f, 'w') as out:
        for row in annotations:
            out.write(f"{row['frame_dir']} {row['label']}\n")


# extract_labels_pose('mmaction2/data/phar/pose/kinesphere_val.pkl')
# -----------------------------------------------------------------------------


def visualize_heatmaps(src_dir='mmaction2/data/phar',
                       ann='resources/annotations/annotations_pose.txt',
                       out_dir='demos/pose',
                       rate=0.1,
                       min_count=30):
    """Visualize a random subset of poses of a dataset.

    Args:
        src_dir (str, optional): videos dir (poses inside this dir).
        ann (str, optional): label map file.
        out_dir (str, optional): out dir to store results.
        rate (float, optional): rate of total poses to be visualized.
        min_count (int, optional): if less than minimal, visualize all
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    script_path = 'tools/demo/visualize_heatmap_volume.py'
    np.random.seed()

    labels = utils.annotations_list(ann)
    for label in tqdm(labels):
        for split in ['train', 'val', 'test']:
            # path to poses dir for a category
            path = osp.join(src_dir, 'pose', split, label)
            pkls = os.listdir(path)
            if not pkls:
                continue

            vis_count = round(len(pkls) * rate)
            if vis_count < min_count:
                # visualize all in case min_count is not reached
                vis_count = len(pkls) if len(pkls) < min_count else min_count

            # pick a vis_count subset to visualize
            to_visualize = np.random.choice(pkls, size=vis_count)
            out = osp.join(out_dir, split, label)
            Path(out).mkdir(parents=True, exist_ok=True)

            for clip in tqdm(to_visualize):
                subargs = [
                    'python',
                    script_path,
                    osp.join(path.replace('pose/', ''),
                             osp.splitext(clip)[0] + '.mp4'),
                    osp.join(path, clip),
                    '--ann',
                    ann,  #
                    '--out-dir',
                    out,  #
                    '--device',
                    'cuda:0'
                ]
                subprocess.run(subargs)


# visualize_heatmaps()
# -----------------------------------------------------------------------------


def merge_pose(path, split, level=2):
    """Given the pose estimation of single videos stored as dictionaries in.

    .pkl format, merge them together and form a list of dictionaries.

    Args:
        path ([string]): path to the pose estimation for individual clips
        split ([string]): train, val, test
    """
    import glob
    import pickle

    result = []
    items = glob.glob(path + '/*' * level + '.pkl')

    for item in items:
        with open(item, 'rb') as f:
            annotations = pickle.load(f)
        result.append(annotations)
    with open(f'kinesphere_{split}.pkl', 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)


# merge_pose('mmaction2/data/phar/pose/val', 'val')
# -----------------------------------------------------------------------------


def filter_pose(path, thr=0.4, correct_rate=0.5, filter_pose=False):
    """Filter Pose estimation based on threshold & correct rate.

    Optionally, if the confidence is less than `thr`, make the corresponding
    pose prediction 0.
    """
    import pickle
    import mmcv

    with open(path, 'rb') as f:
        annotations = pickle.load(f)

    if not isinstance(annotations, list):
        annotations = [annotations]

    CONSOLE.print(f'Processing {len(annotations)} annotations...',
                  style='green')
    new_annotations = []
    for ann in annotations:
        count = 0
        n_person = ann['keypoint'].shape[0]
        for person in range(n_person):
            n_frames = len(ann['keypoint_score'][person])
            for i in range(0, n_frames):
                for j in range(0, 17):
                    if ann['keypoint_score'][person][i][j] < thr:
                        if filter_pose:
                            ann['keypoint'][person][i][j] = 0
                        count += 1
        temp_correct_rate = count / (n_person * n_frames * 17)
        if temp_correct_rate > correct_rate:
            new_annotations.append(ann)

    CONSOLE.print(f'Dumping {len(new_annotations)} annotations.',
                  style='green')
    mmcv.dump(annotations, osp.basename(path))


# filter_pose('mmaction2/data/phar/pose/kinesphere_train.pkl',
#             thr=0.5,
#             correct_rate=0.5)
# -----------------------------------------------------------------------------


def read_pickel(path):
    """Just a method to examine pose information :)

    Args:
        path (_type_): _description_
    """
    import pickle
    with open(path, 'rb') as f:
        annotations = pickle.load(f)

    if type(annotations) is list:
        CONSOLE.print(f'Keys: {annotations[0].keys()}', style='green')
        for ann in annotations:
            CONSOLE.print(ann['label'])
        # CONSOLE.print(annotations[0], style='green')
    else:
        f_no = len(annotations['keypoint'][0])
        pos = int(f_no / 2)
        CONSOLE.print(f'Keys: {annotations.keys()}', style='green')
        CONSOLE.print(f"Label: {annotations['label']}")
        CONSOLE.print(f"Keypoint Shape: {annotations['keypoint'].shape}")
        CONSOLE.print(
            f"Keypoint Score Shape: {annotations['keypoint_score'].shape}")
        # pose estimation
        CONSOLE.print(
            annotations['keypoint'][0][pos],
            style='green')  # keypoint[0] because there is only one person
        # pose estimation confidence
        CONSOLE.print(annotations['keypoint_score'][0][pos], style='green')

        print('\n\n\n')
        print(annotations)


# read_pickel('mmaction2/data/phar/pose/val/69/017S966E.pkl')
# -----------------------------------------------------------------------------


def convert_pose_label(
        path,
        level=2,
        base_ann='resources/annotations/annotations_pose.txt',
        pose_ann='resources/annotations/annotations_pose_new.txt'):
    """Convert pose labels from the all-labels annotations to pose-only
    annotations.

    In the unfortunate case that you generated the pose dataset based on the
    general annotations instead of the pose annotations.
    """

    import glob
    import pickle
    import mmcv
    import utils as utils

    base_ann = utils.annotations_list(base_ann)
    pose_ann = utils.annotations_dic(pose_ann)
    items = glob.glob(path + '/*' * level + '.pkl')

    for item in items:
        with open(item, 'rb') as f:
            annotations = pickle.load(f)
            try:
                annotations['label'] = pose_ann[base_ann[annotations['label']]]
            except KeyError as e:
                CONSOLE.print(item, style='bold red')
                CONSOLE.print(f'KeyError: {e}')
                continue
            mmcv.dump(annotations, item, protocol=pickle.HIGHEST_PROTOCOL)


# convert_pose_label('mmaction2/data/phar/pose/val/')
# -----------------------------------------------------------------------------


def merge_videos(*args):
    """Merge any number of videos using moviepy."""
    import moviepy.editor as mpy
    clips = [mpy.VideoFileClip(clip) for clip in args]
    result = mpy.concatenate_videoclips(clips, method='compose')
    CONSOLE.print('Writing result...', style='green')
    result.write_videofile('concatenated.mp4', audio_codec='aac')


# merge_videos(
#     'mmaction2/data/phar/val/69/0B6U5LKR.mp4',
#     )
# -----------------------------------------------------------------------------


def rewritte_video(video):
    """Simply rewritte a video using OpenCv. Sometimes moviepy writes a video
    with bugs. Simply rewriting it might fix the problem.

    Args:
        video (str): video path
    """
    import cv2

    CONSOLE.print(f'Rewritting {video}', style='green')
    video = cv2.VideoCapture(video)
    video_writer = cv2.VideoWriter(
        'out.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
        video.get(cv2.CAP_PROP_FPS),
        (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
         round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cv2.waitKey(1) < 0:
        success, frame = video.read()
        if not success:
            video.release()
            break

        video_writer.write(frame.astype(np.uint8))


# rewritte_video('It do go duown-DYzT-Pk6Ogw.mkv')
# -----------------------------------------------------------------------------


def resize(file, height=None, width=None, rate=None):
    """Resize a video with moviepy.

    Args:
        file ([type]): [description]
        height ([type], optional): [description]. Defaults to None.
        width ([type], optional): [description]. Defaults to None.
        rate ([type], optional): [description]. Defaults to None.
    """
    import moviepy.editor as mpy
    video = mpy.VideoFileClip(file)
    CONSOLE.print(f'FPS: {video.fps}', style='green')
    CONSOLE.print(f'Width: {video.w} | Height: {video.h}', style='green')
    # portrait video converted to landscape on load
    # https://github.com/Zulko/moviepy/issues/586
    if video.rotation in (90, 270):
        video = video.resize(video.size[::-1])
        video.rotation = 0

    if rate is not None:
        video_resized = video.resize(rate).margin(top=1)
    elif (height is not None) & (width is not None):
        video_resized = video.resize((height, width))
    elif height is not None:
        video_resized = video.resize(height=height)
    elif width is not None:
        video_resized = video.resize(width=width).margin(top=1)

    video_resized.write_videofile(
        osp.split(file)[-1].split('.MOV')[0] + '.mp4')

    CONSOLE.print(
        f'Original size: {video.size}.'
        f'Rescaled to: {video_resized.size}',
        style='green')


# resize('subclip_demo.mp4', width=720)
# -----------------------------------------------------------------------------


def extract_subclip(video, start, finish):
    """Extract subclip from a video using moviepy."""

    import moviepy.editor as mpy
    CONSOLE.print('Writing video...', style='green')

    with mpy.VideoFileClip(video) as v:
        try:
            clip = v.subclip(start, finish)
            clip.write_videofile(f'subclip_{video.split(os.sep)[-1]}',
                                 logger=None,
                                 audio_codec='aac')
        except OSError as e:
            log = (f'! Corrupted Video: {video} | Interval: {start} - {finish}'
                   f'Error: {e}')
            CONSOLE.print(log, style='bold red')


# extract_subclip('dataset_2/602.mp4', 1301, None)
# -----------------------------------------------------------------------------


def extract_frames_from_video(video_path, pos=0, dims=None):
    """Extract frames at a given position of a video using moviepy `dims` is a
    tuple containing width and height."""

    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.resize import resize

    with VideoFileClip(video_path) as video:
        print(f'Video FPS: {video.fps}')
        frame = video.to_ImageClip(pos)
    if dims is not None:
        frame = resize(frame, dims)

    frame.save_frame(f'{pos}_{gen_id(4)}.jpg')


# for i in np.arange(10, 20, 1):
#     extract_frames_from_video('thesis-har/tsn_gradcam.mp4', i)s
# -----------------------------------------------------------------------------


def trim_dataset(path, keep_rate):
    """Remove videos from a directory based on keep_rate.

    Args:
        path (str): path to dataset
        keep_rate (float): % of items to keep
    """
    items = os.listdir(path)
    np.random.seed()
    np.random.shuffle(items)
    items = np.random.choice(items, int(len(items) * keep_rate))
    for item in glob.glob(f'{path}/*'):
        if osp.basename(item) not in items:
            os.remove(item)
    CONSOLE.print(f'Trimmed {path}. {len(items)} videos remaining.',
                  style='green')


# trim_dataset('mmaction2/data/phar/audio/pathhere', 0.14)
# -----------------------------------------------------------------------------


def gen_single_ann_file(path, label, id, splits=['train', 'val'], audio=False):
    """Generate the annotation file for a single class. Id must be given.

    Works for audios so far.

    Args:
        path (str): path to data directory
        label (_type_): label of class
        id (_type_): id to write in annotation file
    """
    for split in splits:
        out = f'{split}.txt'
        with open(out, 'w') as out_f:
            for item in glob.glob(osp.join(path, split, label) + '/*'):
                if audio:
                    count = len(np.load(item))
                    out_f.write(f'{item} {count} {id}\n')
                else:
                    out_f.write(f'{item} {id}\n')
        CONSOLE.print(f'Generated {out}', style='green')


# gen_single_ann_file(path='mmaction2/data/phar/audio_feature/filtered_20/',
#                     label='kissing',
#                     id=4,
#                     audio=True)

# -----------------------------------------------------------------------------


def augment_video(path: str, augment: Callable):
    import cv2

    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    while cv2.waitKey(1) < 0:
        success, frame = video.read()
        if not success:
            video.release()
            break
        frames.append(frame)
    aug_frames = augment(np.array(frames))

    out_f = 'aug.mp4'
    video_writer = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   size)

    for frame in aug_frames:
        video_writer.write(np.array(frame))

    CONSOLE.print(f'Stored result as {out_f}', style='green')


# from vidaug import augmentors as va
# augment_video(path='mmaction2/data/phar/val/69/017S966E.mp4',
#               augment=va.PiecewiseAffineTransform(displacement=2,
#                                 displacement_kernel=1,
#                                 displacement_magnification=2))

# -----------------------------------------------------------------------------


def gen_id(size=8):
    """Generate a random id."""

    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


# -----------------------------------------------------------------------------


def resize_img(img, shape=(480, 480)):
    """Resizes image with moviepy.

    Args:
        img (_type_): _description_
        shape (tuple, optional): _description_. Defaults to (480, 480).
    """
    from moviepy.editor import ImageClip
    ImageClip(img).resize(shape).save_frame(f'{gen_id(4)}.jpg')


# resize('subclip_demo.mp4', (320, 320))
# -----------------------------------------------------------------------------


def download_youtube(link):
    """Downloads a video from Youtube.

    Args:
        link (_type_): _description_
    """
    from pytube import YouTube
    YouTube(link).streams.first().download()


# download_youtube('https://www.youtube.com/watch?v=notLDzBJ2mg&t=66s')
# -----------------------------------------------------------------------------


def extract_timestamps(path):
    """Extract timestamps from csv file based on VIA structure."""
    import pandas as pd

    result = []
    df = pd.read_csv(path)

    for i in range(1, len(df)):
        temp = str(df.iloc[i].value_counts()).split(' ')
        result.append({
            'action':
            temp[0].split(':"')[1].strip('}"'),
            'video':
            ''.join(list(filter(lambda x: x not in '["],', temp[6]))),
            'start':
            float(temp[7][:-1]),
            'end':
            float(temp[8][:-2])
        })

    CONSOLE.print(result)


# extract_timestamps('dataset/265.csv')
# -----------------------------------------------------------------------------


def merge_images_with_font(*args,
                           cols=2,
                           label_rgb=(204, 204, 0),
                           label_pos=(50, 0),
                           font_size=40):
    """Merge two or more images of same size into one based on #cols using
    Pillow True Type Fonts: https://ttfonts.net.

    Used in thesis (in case u need examples)
    """

    from PIL import Image, ImageOps, ImageFont, ImageDraw
    import math

    fnt = ImageFont.truetype('fonts/07558_CenturyGothic.ttf', size=font_size)

    images = []
    for arg in args:
        # open images & add border
        img = ImageOps.expand(Image.open(arg), border=3, fill='blue')
        # add the label to the images
        img = ImageDraw.Draw(img)
        if label_pos is not None:
            img.text(label_pos,
                     arg.split('.')[0].split('/')[-1].split(' ')[0],
                     font=fnt,
                     fill=label_rgb,
                     align='center')
        images.append(img._image)

    size = images[0].size  # (320, 240)
    rows = math.ceil(len(images) / cols)

    # create the new image based on #cols & #rows
    result = Image.new('RGB', (size[0] * cols, size[1] * rows), 'white')

    # add the images to the new image
    c = 0
    for i in range(rows):
        for j in range(cols):
            result.paste(images[c], (j * size[0], i * size[1]))
            c += 1
            if c == len(images):
                break

    result.save('merged_result.jpg')


# merge_images_with_font(
#     'tanet 1.jpg',
#     'tanet 2.jpg',
#     'tanet 3.jpg',
#     'tsn 1.jpg',
#     'tsn 2.jpg',
#     'tsn 3.jpg',
#     cols=3,
#     label_rgb=(0, 255, 255), font_size=20, label_pos=(170, 0))
# -----------------------------------------------------------------------------
