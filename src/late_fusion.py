import argparse
import os.path as osp
import sys

from mmaction.core.evaluation import (get_weighted_score, mean_class_accuracy,
                                      top_k_accuracy)
from mmcv import load
from rich.console import Console
from scipy.special import softmax

sys.path.append('./tools')  # noqa
import utils as utils  # noqa isort:skip

CONSOLE = Console()


def get_class_id(path: str) -> int:
    """Get the label id of a clip given its path (e.g. 1).

    Args:
        path (str): path to clip

    Returns:
        int: label id of clips
    """
    return int(osp.splitext(osp.basename(path.split()[1]))[0])


def get_clip_id(path: str) -> str:
    """Get the name (id) of a clip given its path (e.g. XNXXNAER).

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
    parser.add_argument(
        '--datalists',
        nargs='+',
        help='list of testing data',
        default=[
            'mmaction2/data/phar/val.txt',
            'mmaction2/data/phar/audio_feature/filtered_20/val.txt'
        ])
    parser.add_argument('--apply-softmax', action='store_true')
    parser.add_argument('--top-k',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5],
                        help='top k accuracy to calculate')
    parser.add_argument('--label-map',
                        nargs='+',
                        help='annotation files',
                        default=[
                            'resources/annotations/annotations.txt',
                            'resources/annotations/annotations_audio.txt'
                        ])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.scores) == len(args.coefficients) == len(args.label_map)

    lmaps = []
    for lmap in args.label_map:
        lmaps.append(utils.annotations_dict_rev(lmap))
    score_list = [load(f) for f in args.scores]
    data = [open(dl).readlines() for dl in args.datalists]

    # superset contains all the samples to be tested
    superset = max(data, key=len)
    superset_score = max(score_list, key=len)
    superset_lmap = max(lmaps, key=len)
    # remove the superset from the lists
    i = 0
    while i < len(data):
        if data[i] is superset:
            data.remove(data[i])
            score_list.remove(score_list[i])
            lmaps.remove(lmaps[i])
            break
        i += 1

    # reload superset labels
    superset_lmap = utils.annotations_dic(args.label_map[i])
    labels = [int(x.strip().split()[-1]) for x in superset]
    superset_ids = clip_ids(superset)
    for d in data:
        # CONSOLE.print(set(clip_ids(d)).difference(superset_ids))
        assert set(clip_ids(d)).issubset(superset_ids)

    # order & fill in the scores of the subsets according to the superset
    ordered_scores = []
    superset_ids = clip_ids(superset)
    zeros = [0 for _ in range(len(superset_score[0]))]
    for i in range(len(score_list)):
        ordered_scores.append(list())
        data_ids = clip_ids(data[i])
        for clip in superset:
            id = get_clip_id(clip)
            if id not in data_ids:
                ordered_scores[i].append(zeros)
            else:
                score = score_list[i][data_ids.index(id)]
                to_add = zeros.copy()
                for j in range(len(score)):
                    # add the scores of the models with less classes in the
                    # exact same position as it is in the model that contains
                    # all the classes
                    index = superset_lmap[lmaps[i][j]]
                    to_add[index] = score[j]
                ordered_scores[i].append(to_add)

    ordered_scores.insert(0, superset_score)

    if args.apply_softmax:

        def apply_softmax(scores):
            return [softmax(score) for score in scores]

        ordered_scores = [apply_softmax(scores) for scores in ordered_scores]

    weighted_scores = get_weighted_score(ordered_scores, args.coefficients)
    CONSOLE.print('Weighted Scores', style='green')
    mean_class_acc = mean_class_accuracy(weighted_scores, labels)
    top_k = top_k_accuracy(weighted_scores, labels, args.top_k)
    print(f'Mean Class Accuracy: {mean_class_acc:.04f}')
    for k, topk in enumerate(top_k):
        CONSOLE.print(f'Top {k+1} Accuracy: {topk:.04f}')


if __name__ == '__main__':
    main()
