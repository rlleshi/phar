import glob
import json
import os.path as osp
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

sys.path.append('./src')  # noqa: E501
import utils as utils  # noqa isort:skip

CONSOLE = Console()
ANN_EXT = '.csv'


def get_actions_with_timestamps(path):
    """Given the path to a csv file, get its timestamps.

    The function is specific to the temporal csv annotations produced by the
    VIA annotator.
    """
    results = []
    try:
        df = pd.read_csv(path)
    except Exception:
        CONSOLE.print(f'Error reading {path}', style='error')
    for i in range(1, len(df)):
        temp = str(df.iloc[i].value_counts()).split(' ')
        results.append({
            'action':
            temp[0].split(':"')[1].strip('}"'),
            'video':
            ''.join(list(filter(lambda x: x not in '["],', temp[6]))),
            'start':
            float(temp[7][:-1]),
            'end':
            float(temp[8][:-2])
        })
    return results


def parse_args():
    parser = ArgumentParser(prog='time analysis of annotation distribution '
                            'based on the CSV files annotations.')
    parser.add_argument('--csv-dir',
                        default='dataset/',
                        help='directory of csv annotations')
    parser.add_argument('--out_dir', default='resources/')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations.txt',
                        help='annotation file')
    parser.add_argument('--level',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='directory level of data')
    args = parser.parse_args()
    return args


def save_results(out, result):
    cls = [k for k in result.keys()]
    val = [v for v in result.values()]
    tot = sum(val)
    val = list(map(lambda x: x / tot, val))

    # save json
    result['total'] = tot
    result_json = json.dumps(result, indent=4)
    f = open(osp.join(out, 'annotation_distribution(min).json'), 'w')
    print(result_json, file=f)
    f.close()

    # save plot
    df = pd.DataFrame({'Class': cls, 'Value': val})
    sns.set(rc={'figure.figsize': (15, 13)})
    fig = sns.barplot(x='Class', y='Value', data=df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=15)
    fig.axes.set_title('Sample Distribution / Class ', fontsize=40)
    fig.set_xlabel('Class', fontsize=30)
    fig.set_ylabel('Total %', fontsize=20)
    output = fig.get_figure()
    output.savefig(osp.join(out, 'annotation_distribution.jpg'))


def main():
    args = parse_args()
    ann_count = dict.fromkeys(utils.annotations_dic(args.ann), 0)
    if args.level == 1:
        search = osp.join(args.csv_dir, '*')
    elif args.level == 2:
        search = osp.join(args.csv_dir, '*', '*')
    annotations = [
        item for item in glob.glob(search) if item.endswith(ANN_EXT)
    ]

    for ann in annotations:
        for action in get_actions_with_timestamps(ann):
            label = action['action'].replace('-', '_')
            duration = action['end'] - action['start']
            if np.isnan(duration):
                # faulty annotation
                continue
            try:
                ann_count[label] += duration
            except KeyError:
                CONSOLE.print(f'{ann} has misspelled label {label}',
                              style='yellow')

    ann_count = {k: round(v / 60, 1) for k, v in ann_count.items()}
    save_results(args.out_dir, ann_count)


if __name__ == '__main__':
    main()
