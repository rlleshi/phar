import os
import os.path as osp

from rich.console import Console

CONSOLE = Console()

# Methods:
# 1. long_video_demo
# 2. merge_train_test() - merges train & test sets from the BAST dataset
# 3. read_pickel() - reads and examines pose-estimation pickle files
# 4. merge_pose() - merges individual pose data into a list of dictionaries
#
# 5. extract_frames_from_video() - extract frames of choice from videos
# 6. merge_images_with_font() - merges several images into one
# 7. gen_id() - generate some random id
#
# 8. download_youtube() - download single youtube video
# 9. resize() - resize an image
# 10. demo_posec3d()
# 11. extract subclip() - extract sublicp from video
# 12. extract_timestamps()


def merge_videos(*args):
    """Merge any number of videos."""
    import moviepy.editor as mpy
    clips = [mpy.VideoFileClip(clip) for clip in args]
    result = mpy.concatenate_videoclips(clips, method='compose')
    CONSOLE.print('Writing result...', style='green')
    result.write_videofile('concatenated.mp4', audio_codec='aac')


# merge_videos(
#     'mmaction2/data/phar/val/69/0B6U5LKR.mp4',
#     'mmaction2/data/phar/val/69/7CZ373D8.mp4',
#     'mmaction2/data/phar/val/69/JBZ36GU5.mp4',
#     'mmaction2/data/phar/val/anal/6VC3J40F.mp4',
#     'mmaction2/data/phar/val/anal/8OHAHG9Y.mp4',
#     'mmaction2/data/phar/val/anal/485813HJ.mp4',
#     'mmaction2/data/phar/val/anal/FJS7X9LC.mp4',
#     'mmaction2/data/phar/val/blowjob/0E2A2FFD.mp4',
#     'mmaction2/data/phar/val/blowjob/4KZNUT2G.mp4',
#     'mmaction2/data/phar/val/blowjob/570D27HE.mp4',
#     'mmaction2/data/phar/val/blowjob/6KPG2ANV.mp4',
#     'mmaction2/data/phar/val/cowgirl/0K01XBXN.mp4',
#     'mmaction2/data/phar/val/cowgirl/3KFGRGRB.mp4',
#     'mmaction2/data/phar/val/cowgirl/3PO6E8OY.mp4',
#     'mmaction2/data/phar/val/cowgirl/8WLQY0B8.mp4',
#     'mmaction2/data/phar/val/creampie/14BUOIXZ.mp4',
#     'mmaction2/data/phar/val/creampie/HCEETXQG.mp4',
#     'mmaction2/data/phar/val/creampie/HPSU0L7T.mp4',
#     'mmaction2/data/phar/val/cumshot/0S6M7XKP.mp4',
#     'mmaction2/data/phar/val/cumshot/1J2J0NZM.mp4',
#     'mmaction2/data/phar/val/cumshot/5ZPG3KRC.mp4',
#     'mmaction2/data/phar/val/cunnilingus/0EG1OBFE.mp4',
#     'mmaction2/data/phar/val/cunnilingus/1HO1OZ26.mp4',
#     'mmaction2/data/phar/val/cunnilingus/6UF3OT8N.mp4',
#     'mmaction2/data/phar/val/deepthroat/0C0MQ82H.mp4',
#     'mmaction2/data/phar/val/deepthroat/5C9QXF4E.mp4',
#     'mmaction2/data/phar/val/deepthroat/ASQBFMHL.mp4',
#     'mmaction2/data/phar/val/deepthroat/B3M9WEEQ.mp4',
#     'mmaction2/data/phar/val/doggy/0QWXKEFU.mp4',
#     'mmaction2/data/phar/val/doggy/3TPCR53P.mp4',
#     'mmaction2/data/phar/val/doggy/4N4AMT87.mp4',
#     'mmaction2/data/phar/val/facial_cumshot/0IGDHH6Y.mp4',
#     'mmaction2/data/phar/val/facial_cumshot/1T497O83.mp4',
#     'mmaction2/data/phar/val/facial_cumshot/CJZ8I894.mp4',
#     'mmaction2/data/phar/val/facial_cumshot/HCBMODR7.mp4',
#     'mmaction2/data/phar/val/fingering/0TTEZU3F.mp4',
#     'mmaction2/data/phar/val/fingering/1DDCDOUM.mp4',
#     'mmaction2/data/phar/val/fingering/09Y3SMKW.mp4',
#     'mmaction2/data/phar/val/fondling/0RJ0V1YW.mp4',
#     'mmaction2/data/phar/val/fondling/HMQ6S8PD.mp4',
#     'mmaction2/data/phar/val/fondling/IK9DHY7N.mp4',
#     'mmaction2/data/phar/val/fondling/IYWFQCJA.mp4',
#     'mmaction2/data/phar/val/handjob/0ACBVTAD.mp4',
#     'mmaction2/data/phar/val/handjob/4M5LX4XS.mp4',
#     'mmaction2/data/phar/val/handjob/5CEW1B68.mp4',
#     'mmaction2/data/phar/val/kissing/ZTLD813G.mp4',
#     'mmaction2/data/phar/val/kissing/ZXUH0BNV.mp4',
#     'mmaction2/data/phar/val/kissing/QP2QNG5R.mp4',
#     'mmaction2/data/phar/val/scoop_up/1IAAN8P0.mp4',
#     'mmaction2/data/phar/val/scoop_up/6W37GBZX.mp4',
#     'mmaction2/data/phar/val/scoop_up/FPL82U0N.mp4',
#     'mmaction2/data/phar/val/the_snake/0DTFFLBY.mp4',
#     'mmaction2/data/phar/val/the_snake/6RWW8G0Y.mp4',
#     'mmaction2/data/phar/val/the_snake/7CBJ911N.mp4',
#     'mmaction2/data/phar/val/titjob/1N65GLEE.mp4',
#     'mmaction2/data/phar/val/titjob/2C1I8400.mp4',
#     'mmaction2/data/phar/val/titjob/5JEYD0CL.mp4',
#     )


def extract_timestamps(path):
    """Extract timestamps from csv file based on via structure."""
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
    """Extract subclip from video."""
    import moviepy.editor as mpy
    with mpy.VideoFileClip(video) as v:
        try:
            clip = v.subclip(start, finish)
            CONSOLE.print('Writing video', style='green')
            clip.write_videofile(f'subclip_{video.split(os.sep)[-1]}',
                                 logger=None,
                                 audio_codec='aac')
        except OSError as e:
            log = (f'! Corrupted Video: {video} | Interval: {start} - {finish}'
                   f'Error: {e}')
            CONSOLE.print(log, style='bold red')


# extract_subclip('276.mp4', 190, 600)


# -----------------------------------------------------------------------------
def gen_id(size=8):
    """Generate a random id."""
    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


# -----------------------------------------------------------------------------
def resize_img(img, shape=(480, 480)):
    from moviepy.editor import ImageClip
    ImageClip(img).resize(shape).save_frame(f'{gen_id(4)}.jpg')


# resize('subclip_demo.mp4', (320, 320))


# -----------------------------------------------------------------------------
def download_youtube(link):
    from pytube import YouTube
    YouTube(link).streams.first().download()


# download_youtube('https://www.youtube.com/watch?v=notLDzBJ2mg&t=66s')


# -----------------------------------------------------------------------------
def merge_images_with_font(*args,
                           cols=2,
                           label_rgb=(204, 204, 0),
                           label_pos=(50, 0),
                           font_size=40):
    """Merge two or more images of same size into one based on #cols using
    Pillow True Type Fonts: https://ttfonts.net."""
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
#     extract_frames_from_video('thesis-har/tsn_gradcam.mp4', i)

# -----------------------------------------------------------------------------


def merge_pose(path, split):
    """Given the pose estimation of single videos stored as dictionaries in.

    .pkl format, merge them together and form a list of dictionaries.

    Args:
        path ([string]): path to the pose estimation for individual clips
        split ([string]): train, val, test
    """
    import os
    import os.path as osp
    import pickle
    result = []
    for ann in os.listdir(path):
        if ann.endswith('.pkl'):
            with open(osp.join(path, ann), 'rb') as f:
                annotations = pickle.load(f)
        result.append(annotations)
    with open(f'bast_{split}.pkl', 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)


# merge_pose('minio-transfer/read/pkl', 'train')
# -----------------------------------------------------------------------------


def filter_pose(path, thr=0.5):
    """Filter Pose estimation based on threshold."""
    import pickle
    import mmcv

    with open(path, 'rb') as f:
        annotations = pickle.load(f)

    CONSOLE.print(annotations['keypoint'].shape)

    for person in [0, 1]:
        for i in range(0, len(annotations['keypoint_score'][person])):
            for j in range(0, 17):
                if annotations['keypoint_score'][person][i][j] < thr:
                    CONSOLE.print(annotations['keypoint'][person][i][j])
                    annotations['keypoint'][person][i][j] = 0
                    CONSOLE.print(annotations['keypoint'][person][i][j])

    mmcv.dump(annotations, 'new.pkl')


# filter_pose('demo/pose/blowjob.pkl')

# -----------------------------------------------------------------------------


def read_pickel(path):
    import pickle
    with open(path, 'rb') as f:
        annotations = pickle.load(f)

    if type(annotations) is list:
        CONSOLE.print(f'Keys: {annotations[0].keys()}', style='green')
        CONSOLE.print(annotations[0], style='green')
    else:
        f_no = len(annotations['keypoint'][0])
        pos = int(f_no / 2)
        CONSOLE.print(f'Keys: {annotations.keys()}', style='green')
        # pose estimation
        CONSOLE.print(
            annotations['keypoint'][0][pos],
            style='green')  # keypoint[0] because there is only one person
        # pose estimation confidence
        CONSOLE.print(annotations['keypoint_score'][0][pos], style='green')

        print('\n\n\n')
        print(annotations)


read_pickel('demo/pose/5HE6T27B.pkl')

# -----------------------------------------------------------------------------


# * Merge train & test set FROM BAST dataset
def merge_train_test(path):
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
