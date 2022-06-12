import abc
import argparse
import logging
import os
import os.path as osp
import random as rd
import shutil
import string
import sys
import warnings
from collections import defaultdict

import cv2
import mmcv
import numpy as np
from rich.console import Console

try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for pose extraction.')  # noqa: E501

sys.path.append('src/')  # noqa
import utils as utils  # noqa isort:skip

MMDET_ROOT = 'mmdetection'
MMPOSE_ROOT = 'mmpose'
args = abc.ABC()
args = abc.abstractproperty()
args.det_config = f'{MMDET_ROOT}/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.pose_config = f'{MMPOSE_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501

N_PERSON = 2  # * for bboxes
ANN_TO_INDEX = dict()
CONSOLE = Console()


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def extract_frame(video_path):
    dname = gen_id()
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frame_paths = []
    flag, frame = vid.read()
    first_frame = frame
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    # corrupted video, no frame
    if first_frame is None:
        return None, None

    return frame_paths, first_frame.shape[:2]


def detection_inference(args, frame_paths, det_model=None):
    if det_model is None:
        model = init_detector(args.det_config, args.det_checkpoint,
                              args.device)
    else:
        model = det_model
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    CONSOLE.print('Performing Human Detection for each frame...',
                  style='green')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):
    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, thre=threshold):
        shape = [sum(bbox[:, -1] > thre) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (t - tracklet_st_frame[tlet_id] -
                         len(tracklets[tlet_id]) < 10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] -
                                                   boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(list(range(item.shape[0])),
                          key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def det_postproc(det_results, vid):
    det_results = [removedup(x) for x in det_results]
    CONSOLE.print(f'\nn_person={N_PERSON}', style='green')

    is_easy, bboxes = is_easy_example(det_results, N_PERSON)
    if is_easy:
        msg = f'\n{vid} Easy Example'
        logging.info(msg)
        CONSOLE.print(msg, style='green')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    msg = (f'\n{vid } Hard {N_PERSON}-person Example, '
           f'found {len(tracklets)} tracklet')
    logging.info(msg)
    CONSOLE.print(msg, style='green')

    if N_PERSON == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            # * return np.stack(det_results) - specific to the NTU dataset
            return np.stack(
                np.array([np.array([det_res]) for det_res in det_results]))
        else:
            _, det_results = tracklets2bbox(tracklets, len(det_results))
            return np.array([np.array([det_res]) for det_res in det_results])
            # * return det_results - specific to the NTU dataset

    # * n_person = 2

    if len(tracklets) == 0:
        # no bboxes found at all
        return []

    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference(args, frame_paths, det_results, pose_model=None):
    if pose_model is None:
        model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                args.device)
    else:
        model = pose_model
    CONSOLE.print('Performing Human Pose Estimation for each frame...',
                  style='green')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    num_frame = len(det_results)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp


def pose_extraction(vid,
                    filter_pose,
                    thr=None,
                    det_model=None,
                    pose_model=None):
    frame_paths, img_shape = extract_frame(vid)
    if frame_paths is None and img_shape is None:
        CONSOLE.print(f'{vid} is corrupted', style='red')
        return -1, -1

    det_results = detection_inference(args, frame_paths, det_model)
    det_results = det_postproc(det_results, vid)
    if 0 == len(det_results):
        CONSOLE.print(f'No bounding boxes found for {vid}.', style='yellow')
        return None, None

    pose_results = pose_inference(args, frame_paths, det_results, pose_model)
    anno = dict()
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = img_shape
    anno['original_shape'] = img_shape
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = ANN_TO_INDEX[vid.split('/')[-2]]

    # filter pose estimation based on threshold
    n_person = anno['keypoint_score'].shape[0]
    n_frames = len(anno['keypoint_score'][0])
    count_0 = 0
    for k in range(0, n_person):
        for i in range(0, n_frames):
            for j in range(0, 17):  # 17 defined keypoints
                if anno['keypoint_score'][k][i][j] < thr:
                    if filter_pose:
                        anno['keypoint'][k][i][j] = 0
                    count_0 += 1

    correct_rate = 1 - round(count_0 / (n_person * n_frames * 17), 3)
    CONSOLE.print(
        f'\n{100*correct_rate}% of poses have a threshold higher '
        f'than {thr}',
        style='yellow')
    shutil.rmtree(osp.dirname(frame_paths[0]))

    return anno, correct_rate


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('ann', type=str, help='dataset annotations')
    parser.add_argument('--out-dir',
                        type=str,
                        default='mmaction2/data/phar/pose',
                        help='output dir')
    parser.add_argument('--det-score-thr',
                        type=float,
                        default=0.5,
                        help='detection score threshold')
    parser.add_argument('--pose-score-thr',
                        type=float,
                        default=0.5,
                        help='pose estimation score threshold')
    parser.add_argument('--correct-rate',
                        type=float,
                        default=0.5,
                        help=('if less than this rate of frame poses have a '
                              'lower confidence than `poses-score-thr`, do not'
                              'save the pkl result'))
    parser.add_argument(
        '--filter-pose',
        action='store_true',
        help='whether to set the pose estimation of frames '
        'with score confidence less than the threshold to zero')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args


def main(sub_args, det_model=None, pose_model=None):
    out = osp.join(sub_args.out_dir,
                   osp.splitext(sub_args.video.split('/')[-1])[0]) + '.pkl'
    if osp.exists(out):
        CONSOLE.print(f'{out} exists. Skipping...', style='yellow')
        return

    global ANN_TO_INDEX, args
    args = sub_args
    ANN_TO_INDEX = utils.annotations_dic(args.ann)
    anno, correct_rate = pose_extraction(args.video, args.filter_pose,
                                         args.pose_score_thr, det_model,
                                         pose_model)
    if anno is None and correct_rate is None:
        return 0
    elif anno == -1 and correct_rate == -1:
        return

    # save poses if they don't have more than `args.incorrect_thr %` of poses
    # with a lower confidence than `args.poses_score_thr`
    if correct_rate > args.correct_rate:
        mmcv.dump(anno, out)

    return correct_rate


if __name__ == '__main__':
    logging.basicConfig(filename='pose_extraction.log', level=logging.DEBUG)
    global_args = parse_args()
    args.device = global_args.device
    args.video = global_args.video
    args.out_dir = global_args.out_dir
    args.det_score_thr = global_args.det_score_thr
    args.pose_score_thr = global_args.pose_score_thr
    args.ann = global_args.ann
    args.correct_rate = global_args.correct_rate
    main(args)
