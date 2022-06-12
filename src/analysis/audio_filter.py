import glob
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from rich.console import Console
from tqdm import tqdm

CONSOLE = Console()

EXTS = ['.wav']


def parse_args():
    parser = ArgumentParser(prog='filter audio based on loudness. '
                            'Removes a certain percentile')
    parser.add_argument('src_dir', help='src directory')
    parser.add_argument('out_dir', help='out directory')
    parser.add_argument('--percentile',
                        type=int,
                        default=20,
                        help='thresholding percentile for loudness in db')
    parser.add_argument('--level',
                        type=int,
                        default=1,
                        help='directory level of data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    CONSOLE.print(
        f'Thresholding all audios found in {args.src_dir} with the '
        f'{args.percentile}-th percentile',
        style='green')

    audios = glob.glob(args.src_dir + '/*' * args.level)
    audios = [
        audio for audio in audios if any(audio.endswith(ext) for ext in EXTS)
    ]

    # assuming that all audios have same rate
    _, rate = sf.read(audios[0])
    meter = pyln.Meter(rate)  # meter works with decibels
    loudness = []

    for audio in tqdm(audios):
        data, _ = sf.read(audio)
        loudness.append((audio, meter.integrated_loudness(data)))

    min_db = np.percentile([loud[1] for loud in loudness], args.percentile)
    CONSOLE.print(f'{args.percentile}-th percentile is {min_db}',
                  style='green')

    filtered_audios = list(filter(lambda x: x[1] > min_db, loudness))
    for audio in filtered_audios:
        shutil.copy(audio[0], args.out_dir)


if __name__ == '__main__':
    main()
