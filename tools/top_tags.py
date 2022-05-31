import argparse
import json

from rich.console import Console

CONSOLE = Console()

# top predictions to check for each clip
N = 2


def parse_args():
    parser = argparse.ArgumentParser(description='get the top tags of a video')
    parser.add_argument('predictions', help='json file containing predictions')
    parser.add_argument('--topk',
                        type=int,
                        default=3,
                        choices=[1, 2, 3, 4, 5],
                        help='top k tags to calculate')
    parser.add_argument('--label-map',
                        default='resources/annotations/annotations.txt',
                        help='annotation file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.label_map, 'r') as ann:
        result = {line.strip(): 0 for line in ann}

    assert args.predictions.endswith(
        '.json'), 'prediction file is only supported in json format'
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    for pred in predictions:
        top_pred = list(pred.items())[:N]
        for p in top_pred:
            result[p[1]] += 1

    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    CONSOLE.print(f'Top {args.topk} tags: {list(result.items())[:args.topk]}')


if __name__ == '__main__':
    main()
