import os
import os.path as osp
import pickle
import sys
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import torch
from rich.console import Console
from tqdm import tqdm

from mmaction.apis import inference_recognizer, init_recognizer

# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.append('human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skips

# sys.path.append('/mmaction2/')
CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(prog='accuracy per class for a bunch of clips')
    parser.add_argument('checkpoint', help='model')
    parser.add_argument('split', type=str, help='train/validation/test')
    parser.add_argument(
        '--src-dir',
        type=str,
        default='/mmaction2/data/tanz/videos_val/',
        help='source dir of videos to be evaluated as clips')
    parser.add_argument(
        '--out', type=str, default='/mnt/data_transfer/write/', help='out dir')
    parser.add_argument('--config', type=str, help='model config file')
    parser.add_argument(
        '--ann',
        type=str,
        default=('human-action-recognition/har/annotations/BAST/base/'
                 'tanz_annotations.txt'),
        help='classes/labels')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu/gpu')
    parser.add_argument(
        '--type',
        default='rgb',
        choices=['rgb', 'skeleton'],
        help='rgb or skeleton')
    parser.add_argument(
        '--topk',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        choices=[1, 2, 3, 4, 5],
        help='top-k accuracy to evaluate')
    args = parser.parse_args()
    return args


def save(args, results):
    # if args.out.endswith('.json'):
    # import json
    # results_json = json.dumps(results, indent=4)
    # f = open(out, 'w')
    # print(results , file=f)
    # f.close()
    out = osp.join(args.out, args.split + '_acc_per_class.csv')
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print('Saved {} csv file'.format(out))

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    fig = sns.barplot(x='Class', y='Value', hue='Accuracy', data=df)  #
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30)  #
    fig.axes.set_title('Top 3 Accuracy ' + args.split + '-set', fontsize=40)
    fig.set_xlabel('Class', fontsize=30)
    fig.set_ylabel('Value', fontsize=20)
    output = fig.get_figure()  #
    out = osp.splitext(out)[0] + '.jpg'
    output.savefig(out)
    print('Saved {} plot'.format(out))


def skeleton(
    args,
    number_to_label,
    model,
):
    total_count = helpers.bast_annotations_to_dict(args.ann)
    dist = {k: helpers.bast_annotations_to_dict(args.ann) for k in args.topk}

    for sample in tqdm(os.listdir(args.src_dir)):
        with open(osp.join(args.src_dir, sample), 'rb') as f:
            ann = pickle.load(f)
        ann['start_index'] = 0
        ann['modality'] = 'Pose'
        label = number_to_label[ann['label']]
        total_count[label] += 1
        result = inference_recognizer(model, ann)

        previous_k = 0
        for k in args.topk:
            # if its in top 1 & 2 it will count for top 3
            for i in range(previous_k, k):
                if number_to_label[result[i][0]] == label:
                    dist[k][label] += 1
                    for j in args.topk:
                        # if its in top 3 it will count for top 4
                        if (j != k) & (j > k):
                            dist[j][label] += 1
            previous_k = k

    results = []
    for i in dist.keys():
        for k, v in dist[i].items():
            acc = (v / total_count[k]) if total_count[k] != 0 else 0
            results.append({'Class': k, 'Accuracy': f'acc_{i}', 'Value': acc})
    save(args, results)

    no_labels = 0
    for k in total_count.keys():
        if total_count[k] > 0:
            no_labels += 1
    for i in dist.keys():
        macro_acc = 0
        for k, v in dist[i].items():
            if total_count[k] == 0:
                macro_acc += 0
            else:
                macro_acc += v / total_count[k]

        CONSOLE.print(
            f'Macro top-{i} Acc: '
            f'{round(100 * macro_acc / no_labels, 3)}',
            style='green')


def rgb(args, number_to_label, model):
    total_count = helpers.bast_annotations_to_dict(args.ann)
    dist = {k: helpers.bast_annotations_to_dict(args.ann) for k in args.topk}

    for label in tqdm(os.listdir(args.src_dir)):
        class_dir = osp.join(args.src_dir, label)

        for clip in tqdm(os.listdir(class_dir)):
            previous_k = 0
            total_count[label] += 1
            result = inference_recognizer(model, osp.join(class_dir, clip))

            for k in args.topk:
                # if its in top 1 & 2 it will count for top 3
                for i in range(previous_k, k):
                    if number_to_label[result[i][0]] == label:
                        dist[k][label] += 1
                        for j in args.topk:
                            # if its in top 3 it will count for top 4
                            if (j != k) & (j > k):
                                dist[j][label] += 1
                previous_k = k

    results = []
    for i in dist.keys():
        for k, v in dist[i].items():
            acc = (v / total_count[k]) if total_count[k] != 0 else 0
            results.append({'Class': k, 'Accuracy': f'acc_{i}', 'Value': acc})
    save(args, results)

    no_labels = 0
    for k in total_count.keys():
        if total_count[k] > 0:
            no_labels += 1
    for i in dist.keys():
        macro_acc = 0
        for k, v in dist[i].items():
            if total_count[k] == 0:
                macro_acc += 0
            else:
                macro_acc += v / total_count[k]

        CONSOLE.print(
            f'Macro top-{i} Acc: '
            f'{round(100 * macro_acc / no_labels, 3)}',
            style='green')


def main():
    args = parse_args()
    model = init_recognizer(args.config, args.checkpoint,
                            torch.device(args.device))
    CONSOLE.print(
        f'# Evaluating accuracy per class for the {args.split}-set '
        f'of config file {args.config.split("/")[-1]}',
        style='green')
    number_to_label = helpers.bast_number_to_label(args.ann)
    callback = rgb if args.type == 'rgb' else skeleton
    callback(args, number_to_label, model)


if __name__ == '__main__':
    main()
