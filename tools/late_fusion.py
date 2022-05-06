# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import numpy as np
from mmaction.core.evaluation import (get_weighted_score, mean_class_accuracy,
                                      top_k_accuracy)
from mmcv import load
from rich.console import Console
from scipy.special import softmax

CONSOLE = Console()


def get_clip_id(path: str) -> str:
    """Get the name (id) of a clip given its path.

    Args:
        path (str): path to clip

    Returns:
        str: clip name(id)
    """
    return osp.splitext(osp.basename(path.split()[0]))[0]


def clip_ids(datalist: list) -> list:
    """Returns a list of clip ids given the datalist.

    Args:
        datalist (list): label map

    Returns:
        list: of ids
    """
    return [get_clip_id(d) for d in datalist]


def parse_args():
    parser = argparse.ArgumentParser(description='Fusing multiple scores')
    parser.add_argument('--scores',
                        nargs='+',
                        help='list of scores',
                        default=['demo/fuse/rgb.pkl', 'demo/fuse/flow.pkl'])
    parser.add_argument('--coefficients',
                        nargs='+',
                        type=float,
                        help='coefficients of each score file',
                        default=[1.0, 1.0])
    parser.add_argument('--datalists',
                        nargs='+',
                        help='list of testing data',
                        default=['demo/fuse/data_list.txt'])
    parser.add_argument('--apply-softmax', action='store_true')
    parser.add_argument('--top-k',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5],
                        help='top k accuracy to calculate')
    parser.add_argument(
        '--partial',
        action='store_true',
        help='partial fusion. One list of scores is a subset of the other.'
        'E.g. useful when you are trying to boost only few labels using'
        'an audio-based model.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.scores) == len(args.coefficients)
    score_list = args.scores
    score_list = [load(f) for f in score_list]
    data = [open(dl).readlines() for dl in args.datalists]

    # superset contains all the samples to be tested
    superset = max(data, key=len)
    superset_score = max(score_list, key=len)

    # remove the superset
    i = 0
    while i < len(data):
        if data[i] is superset:
            data.remove(data[i])
            score_list.remove(score_list[i])
            i += 1

    labels = [int(x.strip().split()[-1]) for x in superset]
    superset_ids = clip_ids(superset)
    for d in data:
        assert set(clip_ids(d)).issubset(superset_ids)

    # order & fill in the scores of the subsets according to the superset
    ordered_scores = []
    superset_ids = clip_ids(superset)
    zeros = np.array([0 for _ in range(len(superset_score[0]))])
    for i in range(len(score_list)):
        zeros_residual = np.array([0] * (len(zeros) - len(score_list[i][0])))
        ordered_scores.append(list())
        data_ids = clip_ids(data[i])

        for clip in superset:
            id = get_clip_id(clip)
            if id not in data_ids:
                ordered_scores[i].append(zeros)
            else:
                index = data_ids.index(id)
                ordered_scores[i].append(
                    np.concatenate((score_list[i][index], zeros_residual)))

    ordered_scores.insert(0, superset_score)

    if args.apply_softmax:

        def apply_softmax(scores):
            return [softmax(score) for score in scores]

        ordered_scores = [apply_softmax(scores) for scores in ordered_scores]

    weighted_scores = get_weighted_score(ordered_scores, args.coefficients)
    mean_class_acc = mean_class_accuracy(weighted_scores, labels)
    top_k = top_k_accuracy(weighted_scores, labels, args.top_k)
    print(f'Mean Class Accuracy: {mean_class_acc:.04f}')
    for k, topk in enumerate(top_k):
        CONSOLE.print(f'Top {k+1} Accuracy: {topk:.04f}')


if __name__ == '__main__':
    main()
