import argparse
import os.path as osp
import sys

import cv2
import decord
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from mmpose.apis import vis_pose_result
from mmpose.models import TopDown

from mmaction.datasets.pipelines import Compose

sys.path.append('human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skip

keypoint_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False)
]

limb_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=False,
        with_limb=True)
]

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1


def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30

    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines

    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label

    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)

    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR,
                    THICKNESS, LINETYPE)
    return frame


def vis_skeleton(vid_path, anno, category_name=None, ratio=0.5):
    vid = decord.VideoReader(vid_path)
    frames = [x.asnumpy() for x in vid]

    h, w, _ = frames[0].shape
    new_shape = (int(w * ratio), int(h * ratio))
    frames = [cv2.resize(f, new_shape) for f in frames]

    assert len(frames) == anno['total_frames']
    # The shape is N x T x K x 3
    kps = np.concatenate([anno['keypoint'], anno['keypoint_score'][..., None]],
                         axis=-1)
    kps[..., :2] *= ratio
    # Convert to T x N x K x 3
    kps = kps.transpose([1, 0, 2, 3])
    vis_frames = []

    # we need an instance of TopDown model, so build a minimal one
    model = TopDown(backbone=dict(type='ShuffleNetV1'))

    for f, kp in zip(frames, kps):
        bbox = np.zeros([0, 4], dtype=np.float32)
        result = [dict(bbox=bbox, keypoints=k) for k in kp]
        vis_frame = vis_pose_result(model, f, result)

        if category_name is not None:
            vis_frame = add_label(vis_frame, category_name)

        vis_frames.append(vis_frame)
    return vis_frames


def get_pseudo_heatmap(anno, flag='keypoint'):
    assert flag in ['keypoint', 'limb']
    pipeline = Compose(keypoint_pipeline if flag ==
                       'keypoint' else limb_pipeline)
    return pipeline(anno)['imgs']


def vis_heatmaps(heatmaps, channel=-1, ratio=8):
    # if channel is -1, draw all keypoints / limbs on the same map
    import matplotlib.cm as cm
    h, w, _ = heatmaps[0].shape
    newh, neww = int(h * ratio), int(w * ratio)

    if channel == -1:
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
    cmap = cm.viridis
    heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
    heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
    return heatmaps


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Pose & Heatmap')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('pose_ann', type=str, help='pose pickle annotation')
    parser.add_argument('ann', type=str, help='dataset annotations')
    parser.add_argument(
        '--det-score-thr', type=float, help='detection score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='/mnt/data_transfer/write/')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    anno = load(args.pose_ann)
    categories = helpers.bast_annotations_to_list(args.ann)
    video_name = osp.splitext(args.video.split('/')[-1])[0]

    # visualize skeleton
    vis_frames = vis_skeleton(
        args.video, anno, categories[anno['label']], ratio=1)
    cv2.imwrite(
        osp.join(args.out_dir, f'{video_name}_pose.jpg'),
        vis_frames[int(len(vis_frames) / 2)])
    vid = mpy.ImageSequenceClip(vis_frames, fps=24)
    vid.write_videofile(osp.join(args.out_dir, f'{video_name}_pose.mp4'))

    # visualize heatmaps
    keypoint_heatmap = get_pseudo_heatmap(anno)
    keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
    keypoint_mapvis = [
        add_label(f, categories[anno['label']]) for f in keypoint_mapvis
    ]
    vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
    vid.write_videofile(osp.join(args.out_dir, f'{video_name}_heatmap.mp4'))


if __name__ == '__main__':
    main()
