import argparse
import json
import os
import os.path as osp
import sys

import pandas as pd
import seaborn as sns
from rich.console import Console

sys.path.append('./tools')  # noqa
import utils as utils  # noqa isort:skip

CONSOLE = Console()
PLOT_SPLIT_THR = 26


def parse_args():
    parser = argparse.ArgumentParser(
        description='calculates number of clips / annotation classes')
    parser.add_argument('--src-dir',
                        default='mmaction2/data/phar',
                        help='the dir that contains all the videos')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='the splits where clips are found')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations.txt',
                        help='annotation file')
    parser.add_argument('--out-dir',
                        default='resources/',
                        help='directory to store output files')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    result = utils.annotations_dic(args.ann)
    result = {k: 0 for k, _ in result.items()}

    for split in args.splits:
        path_to_label = osp.join(args.src_dir, split)
        for label in os.listdir(path_to_label):
            result[label] += len(os.listdir(osp.join(path_to_label, label)))

    labels = list(result.keys())
    values = list(result.values())
    result['total'] = sum(values)
    result['average'] = round(result['total'] / len(values))

    # save json
    result_json = json.dumps(result, indent=4)
    f = open(osp.join(args.out_dir, 'ann_dist_clips.json'), 'w')
    print(result_json, file=f)
    f.close()

    # save plot
    dfs = []
    if len(labels) >= PLOT_SPLIT_THR:
        # have to split in at least 2 groups for readability
        dfs.append(
            pd.DataFrame({
                'Class': labels[:int(len(labels) / 2)],
                'Value': values[:int(len(values) / 2)]
            }))
        dfs.append(
            pd.DataFrame({
                'Class': labels[int(len(labels) / 2):],
                'Value': values[int(len(values) / 2):]
            }))
    else:
        dfs.append(pd.DataFrame({'Class': labels, 'Value': values}))

    for df in dfs:
        sns.set(rc={'figure.figsize': (15, 13)})
        fig = sns.barplot(x='Class', y='Value', data=df)
        fig.set_xticklabels(fig.get_xticklabels(), rotation=30)
        fig.axes.set_title('Sample Distribution / Class ', fontsize=40)
        fig.set_xlabel('Class', fontsize=30)
        fig.set_ylabel('Value', fontsize=20)
        output = fig.get_figure()
        output.savefig(
            osp.join(args.out_dir, f'ann_dist_clips_{utils.gen_id(2)}.jpg'))


if __name__ == '__main__':
    main()
