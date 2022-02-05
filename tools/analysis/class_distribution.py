import argparse
import json
import os
import os.path as osp
import sys

import pandas as pd
import seaborn as sns

sys.path.append('/mmaction2/human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skip

EVAL_ANN_NO = 42
groups = {
    'walk-run': [],
    'jump': [],
    'stamp': [],
    'con-contract-expand': [],
    'tiptoe': [],
    'swing': [],
    'spin': [],
    'fall': []
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='calculates number of clips / classes')
    parser.add_argument('training_path', help='path to train/val clips')
    parser.add_argument('testing_path', help='path to test clips')
    parser.add_argument('annotations', help='annotation file')
    parser.add_argument(
        '--type',
        default='general',
        choices=['general', 'group'],
        help='`group` will plots according to the `groups` variable')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    result = helpers.bast_label_to_number_dict(args.annotations)
    result = {k: 0 for k, _ in result.items()}

    for split in [
            osp.join(args.training_path, 'videos_train'),
            osp.join(args.training_path, 'videos_val'), args.testing_path
    ]:
        for label in os.listdir(split):
            result[label] += len(os.listdir(osp.join(split, label)))
    labels = list(result.keys())
    values = list(result.values())
    result['total'] = sum(values)
    result['average'] = round(result['total'] / len(values))

    # save json
    result_json = json.dumps(result, indent=4)
    f = open('cls_dist.json', 'w')
    print(result_json, file=f)
    f.close()

    # save plot
    dfs = []
    if len(labels) == EVAL_ANN_NO:
        if args.type == 'general':
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
            for i in range(len(labels)):
                label = labels[i].split('_')[0]
                for category in groups.keys():
                    if label in category.split('-'):
                        groups[category].append((labels[i], values[i]))

            for _, tup in groups.items():
                dfs.append(
                    pd.DataFrame({
                        'Class': [t[0] for t in tup],
                        'Value': [t[1] for t in tup]
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
        output.savefig(f'cls_dist_{helpers.gen_id(4)}.jpg')


if __name__ == '__main__':
    main()
