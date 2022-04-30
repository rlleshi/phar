import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

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
#
# 2. Video Manipulation
#   2.1 merge_videos(): with MoviePy
#   2.2 resize(): resize videos using MoviePy
#   2.3 extract_subclip(): extract a subclip from a video
#   2.4 extract_frames_from_video(): extract frame from a video
#
# 3. Miscellaneous
#   3.1 gen_id(): generates a 'random' ID
#   3.2 resize_img(): resizes an image using MoviePy
#   3.3 download_youtube(): downloads youtube video
#   3.4 extract_timestamps(): annotation timestamp extraction based on VIA
#   3.5 merge_images_with_font(): merge images and add fonts to them
#   3.6 long_video_demo(): based on MMAction2
#   3.7 demo_posec3d(): based on MMaction2
#
# 4. Obscure
#   4.1 merge_train_test()


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


filter_pose('mmaction2/data/phar/pose/kinesphere_train.pkl',
            thr=0.5,
            correct_rate=0.5)
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


# read_pickel('mmaction2/data/phar/pose/0.4_0.4/val/fondling/0EI0TELM.pkl')
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


# rewritte_video('demo/kinesphere/1s-window/2s-train_48-frames_Kinesphäre_alle_impro_CS.mp4')
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


# extract_subclip('demos/general-test/276.mp4', 190, 600)
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

    fnt = ImageFont.truetype(
        'thesis/scripts-local/fonts/07558_CenturyGothic.ttf', size=font_size)

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
#     'thesis/tanet 1.jpg',
#     'thesis/tanet 2.jpg',
#     'thesis/tanet 3.jpg',
#     'thesis/tsn 1.jpg',
#     'thesis/tsn 2.jpg',
#     'thesis/tsn 3.jpg',
#     cols=3,
#     label_rgb=(0, 255, 255), font_size=20, label_pos=(170, 0))
# -----------------------------------------------------------------------------


def merge_train_test(path):
    """Merge train & test set FROM BAST dataset.

    Args:
        path (_type_): _description_
    """
    import os
    import shutil
    import os.path as osp

    # copy clips
    val_path = osp.join(path, 'videos_val')
    train_path = osp.join(path, 'videos_train')
    for cls in os.listdir(val_path):
        cls_val = osp.join(val_path, cls)
        cls_train = osp.join(train_path, cls)

        for clip in os.listdir(cls_val):
            shutil.move(osp.join(cls_val, clip), cls_train)

    # copy clips list
    with open(osp.join(path, 'tanz_val_list_videos.txt'), 'r') as file:
        val_ann = [line for line in file]
    assert len(val_ann) > 0

    with open(osp.join(path, 'tanz_train_list_videos.txt'), 'a') as file:
        for line in val_ann:
            line.replace('videos_val', 'videos_train')
            file.write(line)

    # restructure
    shutil.rmtree(val_path)
    shutil.rmtree(osp.join(path, 'annotations'))
    os.unlink(osp.join(path, 'tanz_val_list_videos.txt'))
    os.rename(train_path, osp.join(path, 'clips_eval'))
    os.rename(osp.join(path, 'tanz_train_list_videos.txt'),
              osp.join(path, 'tanz_test_list_videos.txt'))

    print('Merged videos_val with videos_train')


# merge_train_test('minio-transfer/read/tanz')
# -----------------------------------------------------------------------------


def long_video_demo():
    import subprocess
    import os
    import random
    from tqdm import tqdm
    out_path = '/mnt/data_transfer/write/avatar_vids/'
    in_path = '/mnt/data_transfer/read/to_process_test/avatar_vid/'
    existing = os.listdir(out_path)

    target = os.listdir(in_path)
    random.shuffle(target)
    for vid in tqdm(target):
        out = vid.split('.')[0] + '.json'
        if out in existing:
            continue
        print(f'Processing {vid}...')
        subargs = [
            'python',
            'human-action-recognition/har/tools/long_video_demo_clips.py',
            os.path.join(in_path, vid),
            ('configs/skeleton/posec3d/'
             'slowonly_r50_u48_240e_ntu120-pr_keypoint_bast.py'),
            ('/mnt/data_transfer_tuning/write/work_dir/8/'
             '56f6783167af4c75835f2021a30bd136/artifacts/'
             'best_top1_acc_epoch_425.pth'),
            os.path.join(out_path,
                         out), '--num-processes', '25', '--num-gpus', '3'
        ]
        subprocess.run(subargs)


# -----------------------------------------------------------------------------


def demo_posec3d(path):
    import subprocess
    import os
    from tqdm import tqdm

    script_path = 'demo/demo_posec3d.py'
    config = ('configs/skeleton/posec3d/'
              'slowonly_r50_u48_240e_ntu120-pr_keypoint_bast.py')
    checkpoint = ('/mnt/data_transfer_tuning/write/work_dir/10/'
                  '4f6aa64c148544198e26bbaf50da2100/artifacts/'
                  'best_top1_acc_epoch_225.pth')
    ann = ('human-action-recognition/har/annotations/BAST/eval/'
           'tanz_annotations_42.txt')

    for clip in tqdm(os.listdir(path)):
        subargs = [
            'python',
            script_path,
            os.path.join(path, clip),
            os.path.join(path, clip),  # overwrite original clip
            '--config',
            config,
            '--checkpoint',
            checkpoint,
            '--label-map',
            ann,  # class annotations
            '--device',
            'cuda:0'
        ]
        subprocess.run(subargs)


# -----------------------------------------------------------------------------
