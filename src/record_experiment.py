import os
import os.path as osp
import re
from argparse import ArgumentParser
from pathlib import Path

import mlflow
from rich.console import Console

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(prog='track experiments with mlflow tracking'
                            'https://mlflow.org/docs/latest/tracking.html')
    parser.add_argument(
        'experiment_name',
        help='name of experiment. Should correspond the model name')
    parser.add_argument(
        'run_name',
        help='name of experiment run. Add things like hyperparameters here.')
    parser.add_argument('work_dir', help='dir where model files are stored')
    parser.add_argument('--mlrun-dir',
                        default='./mlruns',
                        help='mlrun storage dir. Leave default.')
    parser.add_argument('--data-dir',
                        default='mmaction2/data/phar/',
                        help='path to train/val/test dataset')
    args = parser.parse_args()
    return args


def get_train_acc(log, start, topk_length, top_train):
    """Get training accuracy from mmaction2 log files."""
    # * play with these two parameters if the results aren't perfect
    # audio: 1400, 6
    look_back, n_back = 1400, 6
    # train indexes start before needles[1]
    train_index = start
    # take average of last n_back readings
    for row in log[train_index - look_back:train_index].split('\t'):
        for i in range(1, 6):
            t = f'top{i}'
            sub_index = row.find(t)
            if sub_index == -1:
                break

            topk = row[sub_index:sub_index + topk_length]
            topk = float(topk.split('acc: ')[1])
            top_train[t] += topk

    top_train = {k: round(v / n_back, 3) for k, v in top_train.items()}
    return top_train


def get_train_val_acc(logs):
    """Get the validation & training accuracy from mmaction2 log files."""

    # specific to mmaction2 logs
    needles = ('Now best checkpoint is saved as', 'Evaluating top_k_accuracy')
    topk_length = 15

    top_val = {f'top{k}': 0 for k in range(1, 6)}
    top_train = {f'top{k}': 0 for k in range(1, 6)}

    for log in logs:
        # find all indexes for new best models logs
        new_best_indexes = [m.start() for m in re.finditer(needles[0], log)]

        for index in new_best_indexes:
            # topks are replaced only if top1 is exceeded
            replace = False
            # find the start of the new best models log
            start = log[:index].rfind(needles[1])

            for i in range(1, 6):
                t = f'top{i}'
                sub_index = log[start:index].find(t)
                topk = log[start + sub_index:start + sub_index + topk_length]
                topk = float(topk.split('acc')[1])

                if topk > top_val[t] and t == 'top1':
                    replace = True
                if replace:
                    top_val[t] = topk

            if not replace:
                continue

            try:
                top_train = get_train_acc(log, start, topk_length, top_train)
            except IndexError:
                CONSOLE.print('Log is missing train infos', style='yellow')

    return top_train, top_val


def get_last_model(dir):
    """Get the latest checkpoint of a model."""
    latest = osp.join(dir, 'latest.pth')
    if os.path.exists(latest):
        os.remove(osp.join(latest))
    models = [m for m in os.listdir(dir) if m.endswith('.pth')]

    return sorted(models,
                  key=lambda x: int(''.join([d for d in x if d.isdigit()])),
                  reverse=True)


def get_top_model(dir):
    return [model for model in os.listdir(dir) if model[:4] == 'best']


def find_artifact(dir, ext, hint=''):
    """Given a folder, find files based on their extension and part of name."""
    return [
        file for file in os.listdir(dir)
        if (osp.splitext(file)[1] == ext and hint in file)
    ]


def main():
    args = parse_args()
    CONSOLE.print(f'Logging {args.experiment_name}-{args.run_name}...',
                  style='green')
    Path(args.mlrun_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.mlrun_dir)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        logs = []
        # log artifacts from work dir
        for ext in ['.json', '.log', '.py', '.txt', '.pkl']:
            for artifact in find_artifact(args.work_dir, ext):
                mlflow.log_artifact(osp.join(args.work_dir, artifact))
                if ext == '.log':
                    with open(osp.join(args.work_dir, artifact), 'r') as f:
                        logs.append(f.read())

        for ext in ['.txt', '.pkl']:
            for artifact in find_artifact(args.data_dir, ext):
                mlflow.log_artifact(osp.join(args.data_dir, artifact))

        top_model = get_top_model(args.work_dir)
        if not top_model:
            CONSOLE.print(f'No best model found @{args.work_dir}',
                          style='yellow')
        else:
            mlflow.log_artifact(osp.join(args.work_dir, top_model[0]))

        last_model = get_last_model(args.work_dir)
        if not last_model or len(last_model) == 1:
            CONSOLE.print(f'Last saved checkpoint not found @{args.work_dir}',
                          style='yellow')
        else:
            last_model = list(
                filter(lambda x: not x.startswith('best'), last_model))
            mlflow.log_artifact(osp.join(args.work_dir, last_model[0]))

        train_acc, val_acc = get_train_val_acc(logs)

        mlflow.log_params({
            'model': args.experiment_name,
            'run': args.run_name,
            'train acc': f'{train_acc}',
            'val acc': f'{val_acc}',
            'test acc': 'NA'
        })


if __name__ == '__main__':
    main()
