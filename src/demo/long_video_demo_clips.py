import argparse
import os
import os.path as osp
import random
import string
import subprocess
from itertools import repeat
from multiprocessing import Manager, Pool, cpu_count

import moviepy.editor as mpy
import numpy as np
from rich.console import Console

CONSOLE = Console()
manager = Manager()
clips = manager.list()
json_res = manager.list()

MIN_CLIP_DUR = None


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


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
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations_pose.txt',
                        help='for base or eval annotations')
    parser.add_argument('--type',
                        type=str,
                        default='pose',
                        choices=['pose', 'recognition'],
                        help='whether the demo will be pose or recognition')
    parser.add_argument('--num-processes',
                        type=int,
                        default=(cpu_count() - 1 or 1),
                        help='Number of processes to extract subclips')
    parser.add_argument('--num-gpus',
                        type=int,
                        default=1,
                        help='Number of gpus to perform pose-har')
    parser.add_argument('--subclip-duration',
                        type=int,
                        default=7,
                        help='duration of subclips')
    args = parser.parse_args()
    return args


def pose(items):
    gpu, clips, args = items
    script_path = 'src/demo/demo_skeleton.py'
    if not osp.exists(script_path):
        CONSOLE.print(f'{script_path} does not exist', style='red')
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
        result = subprocess.run(subargs, capture_output=True)
        error = result.stderr.decode('utf-8')
        if error:
            CONSOLE.print(error, style='red')


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
    video = mpy.VideoFileClip(video)
    start = timestamps[ts[0]]
    finish = timestamps[ts[1]]

    clip_pth = f'{ts[0]}_{gen_id()}.mp4'
    clips.append(clip_pth)

    try:
        clip = video.subclip(start, finish)
        if clip.duration < MIN_CLIP_DUR:
            CONSOLE.print(f'Subclip duration < {MIN_CLIP_DUR}. Skipping...',
                          style='yellow')
            return
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
            video_clips.append(mpy.VideoFileClip(clip))
        except OSError:
            pass

    result = mpy.concatenate_videoclips(video_clips, method='compose')
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
    global MIN_CLIP_DUR
    MIN_CLIP_DUR = args.subclip_duration

    splits = int(
        mpy.VideoFileClip(args.video).duration / args.subclip_duration)
    timestamps = {
        f'ts{i:02}': args.subclip_duration * i
        for i in range(0, splits + 1)
    }
    time_segments = [(f'ts{i:02}', f'ts{i+1:02}') for i in range(0, splits)]
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
