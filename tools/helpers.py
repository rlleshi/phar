import os
import os.path as osp
import pickle
import random
import string
from argparse import ArgumentTypeError

import torch


def min_max_norm(x):
    """Min-Max normalization using PyTorch."""
    maX = torch.max(x)
    mIn = torch.min(x)
    return (x - mIn) / (maX - mIn)


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def float_parser(val):
    """Must be between range 0 & 1."""
    try:
        val = float(val)
    except ValueError:
        raise ArgumentTypeError(f'{val} must be float')
    if val < 0.0 or val > 1.0:
        raise ArgumentTypeError('Test split out of bounds [0,1]')
    return val


def file_len(file):
    """Return the length of a file."""
    return sum(1 for _ in open(file))


def bast_number_to_label(annotation):
    """Given an annotation file return a dictionary mapping the numbers to
    annotations."""
    result = {}
    with open(annotation) as ann:
        for line in ann:
            (val, key) = line.split(' ')
            result[int(val)] = key.strip().replace('-', '_')
    return result


def bast_label_to_number_dict(annotation):
    """Given an annotation file return a dictionary mapping the annotations to
    numbers."""
    result = {}
    with open(annotation) as ann:
        for line in ann:
            (val, key) = line.split(' ')
            result[key.strip().replace('-', '_')] = int(val)
    return result


def bast_label_to_number(annotation, label):
    """Given an annotation file and a label convert it to its corresponding
    number for the BAST dataset."""
    return bast_label_to_number_dict(annotation).get(label, None)


def bast_annotations_to_list(annotation):
    """Given an annotation file, return a list of them."""
    result = []
    with open(annotation) as ann:
        for line in ann:
            result.append(line.split(' ')[1].strip())
    return result


def bast_annotations_to_dict(annotation):
    """Given an annotation file, return a dictionary counter of them."""
    from collections import Counter
    result = []
    with open(annotation) as ann:
        for line in ann:
            result.append(line.split(' ')[1].replace('-', '_').strip())

    result = Counter(result)
    for k in result.keys():
        result[k] = 0
    return result


def merge_pose_data(in_dir, out_dir, split):
    """Given the pose estimation of single videos stored as dictionaries in.

    .pkl format, merge them together and form a list of dictionaries.

    Args:
        in_dir ([string]): path to the .pkl files for individual clips
        out_dir ([string]): path to the out dir
        split ([string]): train, val, test
    """
    result = []
    for ann in os.listdir(in_dir):
        if ann.endswith('.pkl'):
            with open(osp.join(in_dir, ann), 'rb') as f:
                annotations = pickle.load(f)
        result.append(annotations)

    out_file = osp.join(out_dir, f'bast_{split}.pkl')
    with open(out_file, 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
