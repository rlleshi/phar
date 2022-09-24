import os.path as osp
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
from rich.console import Console

sys.path.append('./tools')  # noqa
import utils as utils  # noqa isort:skip

CONSOLE = Console()
SPEC_EXT = '.npy'


def parse_args():
    parser = ArgumentParser(prog='generate file list for audio dataset based '
                            'on video list')
    parser.add_argument(
        'src_dir',
        type=str,
        help='root dir for video dataset where the ann files are generated')
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='audio_feature',
        help='audio subdir inside the src_dir that contains spectograms')
    parser.add_argument('--split',
                        type=str,
                        nargs='+',
                        default=['train', 'val', 'test'],
                        help='splits where the spectograms are located')
    parser.add_argument('--ann',
                        type=str,
                        default='resources/annotations/annotations_audio.txt',
                        help='audio annotations')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not osp.exists(args.src_dir):
        CONSOLE.print(f'{args.src_dir} not found', style='red')
        return

    ann_to_list = utils.annotations_dic(args.ann)
    for split in args.split:
        split = split + '.txt'
        out_dir = osp.join(args.src_dir, args.audio_dir, split)
        shutil.copyfile(osp.join(args.src_dir, split), out_dir)

        with open(out_dir, 'r') as out:
            content = out.read()

        path = osp.splitext(out_dir)[0]
        with open(out_dir, 'w') as out:

            for line in content.split('\n'):
                if len(line) == 0:
                    continue

                _, category, clip = line.rsplit(osp.sep, 2)
                new_path = osp.join(path, category, clip).split(' ')[0]
                new_class_id = ann_to_list.get(category, None)

                if new_class_id is not None:
                    new_path = new_path.split('.')[0] + SPEC_EXT
                    if not osp.exists(new_path):
                        # corresponding .npy file doesn't exist (e.g. filtered)
                        continue

                    count = len(np.load(new_path))
                    out.write(f'{new_path} {count} {new_class_id}\n')

        CONSOLE.print(f'Created list file @{out_dir}', style='green')


if __name__ == '__main__':
    main()
