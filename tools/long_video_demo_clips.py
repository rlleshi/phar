import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from itertools import repeat
from multiprocessing import Manager, Pool, cpu_count

import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from rich.console import Console

sys.path.append('human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skips

CONSOLE = Console()
manager = Manager()
clips = manager.list()
json_res = manager.list()


def prettify(byte_content):
    decoded = byte_content.decode('utf-8')
    formatted_output = decoded.replace('\\n', '\n').replace('\\t', '\t')
    return formatted_output


def delete_clips(clips):
    for clip in clips:
        try:
            os.unlink(clip)
        except FileNotFoundError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='long video demo based on clips')
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='model checkpoint')
    parser.add_argument('out', help='out file. Video or Json')
    parser.add_argument(
        '--ann-bast',
        type=str,
        help='.eaf bast annotation file with timestamps')
    parser.add_argument(
        '--ann',
        type=str,
        default=('human-action-recognition/har/annotations/BAST/base/'
                 'tanz_annotations.txt'),
        help='for base or eval annotations')
    parser.add_argument(
        '--type',
        type=str,
        default='pose',
        choices=['pose', 'recognition'],
        help='whether the demo will be pose or recognition')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=(cpu_count() - 1 or 1),
        help='Number of processes to extract subclips')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=4,
        help='Number of gpus to perform pose-har')
    args = parser.parse_args()
    return args


def get_time_segments(args):
    tree = ET.parse(args.ann_bast)
    root = tree.getroot()

    # {'ts1': '3320', ..., 'ts58': '235798'}
    timestamps = {}
    for ts in root.iter('TIME_SLOT'):
        id = ts.attrib['TIME_SLOT_ID']
        if len(id) == 3:
            id = f'{id[0:2]}0{id[2:]}'
        timestamps[id] = ts.attrib['TIME_VALUE']

    # [('ts1', 'ts3'}, ..., ('ts53', 'ts56')]
    time_segments = []
    ids = ['a' + str(i) for i in range(0, 10)]
    for base_annotation in root.iter('ALIGNABLE_ANNOTATION'):
        if base_annotation.attrib['ANNOTATION_ID'] not in ids:
            continue
        start_ts = base_annotation.attrib['TIME_SLOT_REF1']
        end_ts = base_annotation.attrib['TIME_SLOT_REF2']
        if len(start_ts) == 3:
            start_ts = f'{start_ts[0:2]}0{start_ts[2:]}'
        if len(end_ts) == 3:
            end_ts = f'{end_ts[0:2]}0{end_ts[2:]}'
        time_segments.append((start_ts, end_ts))

    return timestamps, time_segments


def pose(items):
    gpu, clips, args = items
    script_path = 'demo/demo_posec3d.py'
    for clip in clips:
        subargs = [
            'python',
            script_path,
            clip,
            clip,  # overwrite original clip
            '--config',
            args.config,
            '--checkpoint',
            args.checkpoint,
            '--label-map',
            args.ann,  # class annotations
            '--device',
            gpu
        ]
        try:
            result = subprocess.run(subargs, capture_output=True)
            json_res.append(clip[2:4] + '-' +
                            prettify(result.stdout).split('result')[1])
        except Exception as e:
            CONSOLE.print(e, style='bold red')


def recognition(items):
    gpu, clips, args = items
    script_path = 'demo/demo.py'
    for clip in clips:
        subargs = [
            'python',
            script_path,
            args.config,
            args.checkpoint,
            clip,
            args.ann,  # class annotations
            '--font-color',
            'blue',
            '--out-filename',
            clip,  # overwrite original clip
            '--device',
            gpu
        ]
        try:
            subprocess.check_output(subargs)
        except Exception as e:
            CONSOLE.print(e, style='bold red')


def extract_subclip(items):
    ts, timestamps, video = items
    video = VideoFileClip(video)
    start = float(timestamps[ts[0]]) / 1000
    try:
        finish = float(timestamps[ts[1]]) / 1000
    except TypeError:
        finish = timestamps[ts[1]]

    clip_pth = f'{ts[0]}_{helpers.gen_id(4)}.mp4'
    clips.append(clip_pth)

    clip = video.subclip(start, finish)
    try:
        clip.write_videofile(clip_pth, logger=None, audio=False)
    except OSError as e:
        CONSOLE.print(e, style='bold red')
        pass
    finally:
        video.close()


def merge_clips(clips, out):
    clips = sorted(clips, key=lambda x: int(x[2:4]))
    video_clips = []
    for clip in clips:
        try:
            video_clips.append(VideoFileClip(clip))
        except OSError:
            pass
    result = concatenate_videoclips(video_clips)
    result.write_videofile(out)
    delete_clips(clips)


def merge_json(json_res, time_segments, out):
    result = {}
    json_res = sorted(json_res, key=lambda x: int(x[:2]))
    for tup in zip(time_segments, json_res):
        result[str(tup[0])] = tup[1].split(' ', 1)[1].strip()

    import json
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    args = parse_args()
    if args.ann_bast is not None:
        timestamps, time_segments = get_time_segments(args)
    else:
        # if no annotations, predict for 10s segments
        splits = int(VideoFileClip(args.video).duration / 10)
        timestamps = {f'ts{i:02}': 10000 * i for i in range(0, splits + 1)}
        time_segments = [(f'ts{i:02}', f'ts{i+1:02}')
                         for i in range(0, splits)]
        # add a timestamp for any remaining segments < 10s
        rest_timestamp = f'ts{int(list(timestamps.keys())[-1][2:]) + 1}'
        timestamps[rest_timestamp] = None
        time_segments.append(
            (list(timestamps.keys())[-2], list(timestamps.keys())[-1]))

    CONSOLE.print('Extracting subclips...', style='green')
    pool1 = Pool(args.num_processes)
    gpus = [f'cuda:{i}' for i in range(args.num_gpus)]
    pool1.map(extract_subclip,
              zip(time_segments, repeat(timestamps), repeat(args.video)))

    pool2 = Pool(len(gpus))
    callback = pose if args.type == 'pose' else recognition
    CONSOLE.print(f'Performing {args.type}...', style='green')
    clips_per_gpus = [
        label_split for label_split in np.array_split(clips, args.num_gpus)
    ]
    pool2.map(callback, zip(gpus, clips_per_gpus, repeat(args)))

    merge_clips(clips, args.out.split('.')[0] + '.mp4')
    if args.out.endswith('.json'):
        merge_json(json_res, time_segments, args.out)


if __name__ == '__main__':
    main()
