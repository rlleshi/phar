import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from itertools import repeat
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import cv2
import mmcv
import moviepy.editor as mpy
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import yaml
from demo.demo_skeleton import frame_extraction
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.core.evaluation import get_weighted_score
from rich.console import Console
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

sys.path.append('./mmaction2')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.85
FONTCOLOR = (255, 255, 0)  # BGR, white
FONTCOLOR_SCORE = (0, 165, 255)
THICKNESS = 1
LINETYPE = 1
AUDIO_FEATURE_SCRIPT = 'mmaction2/tools/data/build_audio_features.py'
LOUD_WEIGHT = None
TEMP = 'temp'
x, y = 10, 30
x_y_dist = 200
placeholder = {
    'kissing': 0,
    'fondling': 0,
    'handjob': 0,
    'fingering': 0,
    'titjob': 0
}
num_keypoint = 17
sk_thr = 3
a_thr = 2
CONSOLE = Console()
manager = Manager()
clips = manager.list()
PREDS = {}
USED_MODS = {}  # used models were used per clip prediction


def _delete():
    shutil.rmtree(TEMP, ignore_errors=True)
    shutil.rmtree('./tmp', ignore_errors=True)


def _extract_clip(items):
    """Extract a particular clip."""
    ts, video = items
    key = next(iter(ts))
    start, finish = next(iter(ts.values()))
    if finish - start < 2:
        return
    video = mpy.VideoFileClip(video)
    clip_pth = osp.join(TEMP, f'{key}_{start}_{finish}.mp4')
    clips.append(clip_pth)
    try:
        clip = video.subclip(start, finish)
        clip.write_videofile(clip_pth, logger=None, audio=True)
    except OSError as e:
        CONSOLE.print(e, style='bold red')
        pass


def extract_clips(video: str, s_len: int, num_proc: int):
    """Extract clips given a video based on a sliding window.

    Args:
        video (str): video
        s_len (int): subclip length (sliding window)
        num_proc (int): num of processes
    """
    video_dur = mpy.VideoFileClip(video).duration
    splits, remainder = int(video_dur / s_len), int(video_dur % s_len)
    CONSOLE.print(f'Extracting {splits} sublicps for {video}...',
                  style='green')
    time_segments = [{
        f'ts{i:02}': [s_len * i, s_len * i + s_len]
    } for i in range(0, splits)]
    last = {
        f'ts{int(list(time_segments[-1])[0][2:]) + 1}':
        [video_dur - int(remainder), video_dur]
    }
    time_segments.append(last)
    pool = Pool(num_proc)
    pool.map(_extract_clip, zip(time_segments, repeat(video)))


def detection_inference(det_score_thr, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        det_score_thr (float): bbox threshold
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    assert DET_MODEL.CLASSES[0] == 'person', ('Please use a detector trained '
                                              'on COCO')
    results = []
    CONSOLE.print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(DET_MODEL, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(frame_paths, det_results):
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(POSE_MODEL, f, d,
                                             format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def cleanup(original_video, tmp_out_video, out_video):
    """Add original audio to demo & cleanup."""
    # extract & add audio to demo
    cmd_extract = (f'ffmpeg -i {original_video} -f mp3 -ab 192000 -vn '
                   'audio.mp3 -y')
    os.popen(cmd_extract)
    time.sleep(5)
    cmd_add = (f'ffmpeg -i {tmp_out_video} -i audio.mp3 -c copy -map 0:v:0 '
               f'-map 1:a:0 {out_video} -y')
    os.popen(cmd_add)
    time.sleep(5)

    # cleanup
    os.remove('audio.mp3')
    os.remove(tmp_out_video)


def parse_args():
    parser = argparse.ArgumentParser(
        description='image/pose/audio based inference')
    parser.add_argument('video', help='video file')
    parser.add_argument('out', help='out file. Video or Json')
    parser.add_argument(
        '--label-maps',
        type=str,
        nargs='+',
        default=[
            'resources/annotations/annotations.txt',
            'resources/annotations/annotations_pose.txt',
            'resources/annotations/annotations_audio.txt'
        ],
        help='labels for rgb/pose/audio based action recognition models')
    parser.add_argument(
        '--rgb-config',
        default=('mlruns/7/d287e786e6a04ceba5c6a644a5be7ebf/artifacts/'
                 'timesformer_divST_8x32x1_15e_kinetics400_rgb.py'),
        help='rgb-based action recognizer config file')
    parser.add_argument(
        '--rgb-checkpoint',
        # TODO: replace with git coordinates?
        default=('./mlruns/7/d287e786e6a04ceba5c6a644a5be7ebf/artifacts/'
                 'best_top1_acc_epoch_13.pth'),
        help='rgb-based action recognizer model checkpoint')
    parser.add_argument(
        '--skeleton-config',
        default=('mlruns/3/3d6c9a9a7bc043479ef3cc168d3eb2ad/artifacts/'
                 'slowonly_r50_u54_480e_pr-kinetics.py'),
        help='skeleton-based action recognizer config file')
    parser.add_argument(
        '--skeleton-checkpoint',
        default=('mlruns/3/3d6c9a9a7bc043479ef3cc168d3eb2ad/artifacts/'
                 'best_top1_acc_epoch_395.pth'),
        help='pose-based action recognizer model checkpoint')
    parser.add_argument(
        '--audio-config',
        default=('mlruns/4/bb25d9cab8cf46f48716c3b3b29f5fd9/artifacts/'
                 'audioonly_r50_64x1x1_100e_audio_feature.py'),
        help='audio-based action recognizer config file')
    parser.add_argument(
        '--audio-checkpoint',
        default=('mlruns/4/bb25d9cab8cf46f48716c3b3b29f5fd9/artifacts/'
                 'best_top1_acc_epoch_55.pth'),
        help='audio-based action recognizer model checkpoint')
    parser.add_argument(
        '--pose-config',
        default='mmaction2/demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url (used for skeleton)')
    parser.add_argument(
        '--det-config',
        default='mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection model checkpoint (used for skeleton')
    parser.add_argument('--det-score-thr',
                        type=float,
                        default=0.8,
                        help='the threshold of human detection score')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=(cpu_count() - 1 or 1),
        help='Number of processes to extract subclips from the video')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    parser.add_argument('--subclip-len',
                        type=int,
                        default=7,
                        help='duration of subclips. Sliding window.')
    parser.add_argument('--short-side',
                        type=int,
                        default=480,
                        help='specify the short-side length of the image')
    parser.add_argument('--coefficients',
                        nargs='+',
                        type=float,
                        help='coefficients of each model (rgb, skelet, audio)',
                        default=[0.4, 0.8, 1.0])
    parser.add_argument('--pose-score-thr',
                        type=float,
                        default=0.4,
                        help=('pose estimation score threshold. Not all videos'
                              ' are suitable for skeleton-based HAR'))
    parser.add_argument(
        '--correct-rate',
        type=float,
        default=0.35,
        help=('if less than this rate of frame poses have a '
              'lower confidence than `poses-score-thr`, skip the demo'))
    parser.add_argument(
        '--loudness-weights',
        type=str,
        default='resources/audio/db_20_config.yml',
        help=('audio loudness thresholds for each class. Not all videos have '
              'loud enough audio suitable for audio-based HAR'))
    parser.add_argument('--topk', type=int, default=5, help='top k accuracy')
    parser.add_argument('--timestamps',
                        action='store_true',
                        help='generate timestamps')
    args = parser.parse_args()
    return args


def rgb_inference():
    """Perform RGB-model inference."""
    global PREDS, USED_MODS
    for clip in tqdm(clips):
        PREDS[clip] = {'rgb': {}}
        USED_MODS[clip] = [0]
        results = inference_recognizer(RGB_MODEL, clip)
        results = [(RGB_LABELS[r[0]], r[1]) for r in results]
        for r in results:
            PREDS[clip]['rgb'][r[0]] = r[1]


def skeleton_inference(clip: str, args: dict):
    """Perform skeleton-model inference.

    Args:
        clip (str): video
        args (dict): parsed args
    """
    global PREDS, USED_MODS
    if set(list(PREDS[clip]['rgb'].keys())[:sk_thr]).isdisjoint(POSE_LABELS):
        CONSOLE.print(
            f'Skipping {clip} for skeleton inference. Labels were'
            f' not found in top {sk_thr} preds of the rgb model',
            style='yellow')
        PREDS[clip]['pose'] = placeholder
        return

    frame_paths, original_frames = frame_extraction(clip, args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape
    det_results = detection_inference(args.det_score_thr, frame_paths)
    torch.cuda.empty_cache()
    pose_results = pose_inference(frame_paths, det_results)
    torch.cuda.empty_cache()

    data = dict(frame_dir='',
                label=-1,
                img_shape=(h, w),
                original_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    data['keypoint'] = keypoint
    data['keypoint_score'] = keypoint_score

    tmp_frame_dir = osp.dirname(frame_paths[0])
    count_0 = 0
    for k in range(0, num_person):
        for i in range(0, num_frame):
            for j in range(0, 17):  # 17 defined keypoints
                if data['keypoint_score'][k][i][j] < args.pose_score_thr:
                    # fake_anno['keypoint'][k][i][j] = 0
                    count_0 += 1
    try:
        correct_rate = 1 - round(count_0 / (num_person * num_frame * 17), 3)
    except ZeroDivisionError:
        correct_rate = 0
    if correct_rate < args.correct_rate:
        CONSOLE.print((f'Clip has correct rate of {correct_rate}, lower than '
                       f'the threshold of {args.correct_rate}. Skipping...'),
                      style='red')
        shutil.rmtree(tmp_frame_dir)
        PREDS[clip]['pose'] = placeholder
        return

    results = inference_recognizer(SK_MODEL, data)
    PREDS[clip]['pose'] = {}
    results = [(POSE_LABELS[r[0]], r[1]) for r in results]
    for r in results:
        PREDS[clip]['pose'][r[0]] = r[1]
    shutil.rmtree(tmp_frame_dir)
    USED_MODS[clip].append(1)


def audio_inference(clip: str):
    """Audio based action recognition."""
    global PREDS, USED_MODS
    if set(list(PREDS[clip]['rgb'].keys())[:a_thr]).isdisjoint(AUDIO_LABELS):
        CONSOLE.print(f'Skipping {clip} for audio inference...',
                      style='yellow')
        PREDS[clip]['audio'] = placeholder
        return

    CONSOLE.print(f'Extracting audio for {clip}...', style='green')
    out_audio = f'{osp.splitext(clip)[0]}.wav'
    cmd = f'ffmpeg -i ./{clip}  -map 0:a  -y {out_audio}'
    os.popen(cmd)
    time.sleep(1)

    data, rate = sf.read(out_audio)
    meter = pyln.Meter(rate)  # meter works with decibels
    if meter.integrated_loudness(data) < LOUD_WEIGHT:
        CONSOLE.print(f'Audio for clip {clip} is not loud enough. Skipping...',
                      style='yellow')
        PREDS[clip]['audio'] = placeholder
        return

    CONSOLE.print(f'Building audio features for {clip}...', style='green')
    out_feature = f'{osp.splitext(out_audio)[0]}.npy'
    subargs = ['python', AUDIO_FEATURE_SCRIPT, TEMP, TEMP, '--ext', '.wav']
    subprocess.run(subargs)

    results = inference_recognizer(AUDIO_MODEL, out_feature)
    results = [(AUDIO_LABELS[k[0]], k[1]) for k in results]
    PREDS[clip]['audio'] = {}
    for r in results:
        PREDS[clip]['audio'][r[0]] = r[1]

    os.remove(out_audio)
    os.remove(out_feature)
    USED_MODS[clip].append(2)


def get_weighted_scores(clip: str, coeffs: list) -> dict:
    """Get the weighted scores of all modules sorted in descending order.

    Args:
        clip (str): current clip
        coeffs (list): weight coefficients

    Returns:
        dict: weighted scores
    """
    scores = []
    for module in PREDS[clip]:
        score = [0 for _ in range(len(RGB_LABELS))]
        for k in PREDS[clip][module]:
            score[RGB_LABELS.index(k)] = PREDS[clip][module][k]
        scores.append(score)
    weighted_scores = get_weighted_score(scores, coeffs)

    result = {}
    for i in range(len(weighted_scores)):
        if 0 == weighted_scores[i]:
            continue
        result[weighted_scores[i]] = RGB_LABELS[i]
    return dict(sorted(result.items(), reverse=True))


def write_results_video(args: dict):
    """Write the results to a video demo.

    Args:
        args (dict): parsed args
    """
    tmp_out_video = f'tmp_{osp.splitext(args.out)[0]}.mp4'
    video = cv2.VideoCapture(args.video)
    video_writer = cv2.VideoWriter(
        tmp_out_video, cv2.VideoWriter_fourcc(*'MP4V'),
        video.get(cv2.CAP_PROP_FPS),
        (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
         round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    CONSOLE.print('Writing results to video...', style='bold green')

    for c in PREDS:
        # extract frames from clip
        clip = cv2.VideoCapture(c)
        frames = []
        while cv2.waitKey(1) < 0:
            success, frame = clip.read()
            if not success:
                clip.release()
                break
            frames.append(frame)

        result = get_weighted_scores(c, args.coefficients)

        for frame in frames:
            i = 1
            for topk in result:
                if i == args.topk:
                    break
                # scale the score depending on coeffs and #models used
                n_mods = len(USED_MODS[c])
                score_scaler = n_mods / sum(
                    [args.coefficients[m] for m in USED_MODS[c]])
                score = round(topk / n_mods, 2) * score_scaler

                cv2.putText(frame, result[topk], (x, y * i), FONTFACE,
                            FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
                cv2.putText(frame, str(round(100 * score, 2)),
                            (x + x_y_dist, y * i), FONTFACE, FONTSCALE,
                            FONTCOLOR_SCORE, THICKNESS, LINETYPE)
                i += 1
            video_writer.write(frame.astype(np.uint8))
    video_writer.release()
    cleanup(args.original_video, tmp_out_video, args.out)


def write_results_json(args: dict):
    """Write results to json.

    Args:
        args (dict): _description_
    """
    results = []
    for c in PREDS:
        results.append(get_weighted_scores(c, args.coefficients))
    with open(args.out, 'w') as js:
        json.dump(results, js)


def write_timestamps(args: dict):
    """Write video timestamps.

    Args:
        args (dict): _description_
    """
    results = {}
    i = 0
    for c in PREDS:
        topks = get_weighted_scores(c, args.coefficients)
        start = i * args.subclip_len
        end = start + args.subclip_len
        results[f'{start}:{end}'] = list(topks.items())[0][1]
        i += 1
    with open(f'{osp.splitext(args.out)[0]}_ts.json', 'w') as js:
        json.dump(results, js)


# TODO: performance improvements: multi GPU processing for rgb, skeleton
# (ggf. detection & pose estimation), and finally for audio
# TODO: refactoring is needed
def main():
    args = parse_args()
    global RGB_LABELS, POSE_LABELS, AUDIO_LABELS
    RGB_LABELS, POSE_LABELS, AUDIO_LABELS = [
        [x.strip() for x in open(path).readlines()] for path in args.label_maps
    ]
    Path(TEMP).mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # resize video to make inferences faster
    CONSOLE.print('Resizing video...', style='green')
    video = mpy.VideoFileClip(args.video)
    video_resized = video.resize(height=480)
    out_video = osp.join(TEMP, osp.basename(args.video))
    video_resized.write_videofile(out_video)
    args.original_video = args.video
    args.video = out_video

    extract_clips(args.video, args.subclip_len, args.num_processes)

    global RGB_MODEL, SK_MODEL, AUDIO_MODEL, DET_MODEL, POSE_MODEL
    RGB_MODEL = init_recognizer(args.rgb_config,
                                args.rgb_checkpoint,
                                device=args.device)
    CONSOLE.print(
        f'Finished in {round((time.time() - start_time) / 60, 2)} min',
        style='green')
    CONSOLE.print('Performing RGB inference...', style='green')
    rgb_inference()
    CONSOLE.print(
        f'Finished in {round((time.time() - start_time) / 60, 2)} min',
        style='green')
    torch.cuda.empty_cache()

    global PREDS
    if args.skeleton_checkpoint:
        DET_MODEL = init_detector(args.det_config, args.det_checkpoint,
                                  args.device)
        POSE_MODEL = init_pose_model(args.pose_config, args.pose_checkpoint,
                                     args.device)
        SK_MODEL = init_recognizer(args.skeleton_config,
                                   args.skeleton_checkpoint, args.device)

        CONSOLE.print('Performing skeleton inference...', style='green')
        for clip in tqdm(PREDS):
            skeleton_inference(clip, args)
        CONSOLE.print(
            f'Finished in {round((time.time() - start_time) / 60, 2)} min',
            style='green')
        torch.cuda.empty_cache()

    if args.audio_checkpoint:
        AUDIO_MODEL = init_recognizer(args.audio_config,
                                      args.audio_checkpoint,
                                      device=args.device)
        global LOUD_WEIGHT
        loud_weights = yaml.load(open(args.loudness_weights, 'r'),
                                 Loader=yaml.FullLoader)
        LOUD_WEIGHT = sum(loud_weights.values()) / len(loud_weights)

        CONSOLE.print('Performing audio inference...', style='green')
        for clip in tqdm(PREDS):
            audio_inference(clip)
        torch.cuda.empty_cache()

    CONSOLE.print(
        f'Finished in {round((time.time() - start_time) / 60, 2)} min',
        style='green')
    PREDS = dict(sorted(PREDS.items()))
    CONSOLE.print(PREDS)

    if args.out.endswith('.json'):
        write_results_json(args)
    else:
        write_results_video(args)
    if args.timestamps:
        write_timestamps(args)
    _delete()

    CONSOLE.print(
        f'Finished in {round((time.time() - start_time) / 60, 2)} min',
        style='green')


if __name__ == '__main__':
    main()
