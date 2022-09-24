import os
import os.path as osp
import subprocess

import schedule
from rich.console import Console

CONSOLE = Console()

# https://schedule.readthedocs.io/en/stable/examples.html


def pose_feasibility(cat, out_dir='mmaction2/data/phar/pose'):
    """Schedule for the pose_feasibility.py script."""
    CONSOLE.print(f'Checking pose feasibility for {cat}...', style='green')
    script_path = 'tools/analysis/pose_feasibility.py'

    subargs = ['python', script_path, cat, '--out-dir', out_dir, '--resume']
    subprocess.run(subargs)
    return schedule.CancelJob


def extract_audio(in_dir, out_dir):
    """Scheduler to extract audio from videos_val.

    Args:
        in_dir (_type_): _description_
        out_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    import time
    script_dir = '/mmaction2/tools/data/extract_audio.py'
    for dIr in os.listdir(in_dir):
        CONSOLE.print(f'Extracting videos for {dIr}...', style='green')
        CONSOLE.print(osp.join(in_dir, dIr))
        CONSOLE.print(osp.join(out_dir, dIr))

        subargs = [
            'python', script_dir,
            osp.join(in_dir, dIr),
            osp.join(out_dir, dIr), '--level', '1', '--ext', 'avi'
        ]
        subprocess.run(subargs)
        time.sleep(30)
    return schedule.CancelJob


def extract_audio_feature(in_dir, out_dir):
    """Extract spectogram features from audio.

    Args:
        in_dir (_type_): _description_
        out_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    script_dir = '/mmaction2/tools/data/build_audio_features.py'
    for dIr in os.listdir(in_dir):
        dir_path = osp.join(in_dir, dIr)
        for audio in os.listdir(dir_path):
            audio_path = osp.join(dir_path, audio)
            subargs = [
                'python', script_dir, audio_path,
                osp.join(out_dir,
                         audio.split('.')[0] + '.npy'), '--level', '1',
                '--ext', 'avi'
            ]
            subprocess.run(subargs)

    return schedule.CancelJob


def train_model(config: str,
                work_dir: str,
                resume_from=None,
                cfg_options=None):
    script_path = 'mmaction2/tools/dist_train.sh'
    no_gpus = 1
    subargs = [
        'bash', script_path, config,
        str(no_gpus), '--work-dir', work_dir, '--validate'
    ]
    if resume_from:
        subargs.append('--resume-from')
        subargs.append(resume_from)
    if cfg_options:
        subargs.append('--cfg-options')
        for tup in cfg_options.items():
            subargs.append(f'{tup[0]}={tup[1]}')
    subprocess.run(subargs)


def demo(in_video, out_video):
    script_path = 'src/demo/multimodial_demo.py'
    subargs = ['python', script_path, in_video, out_video]
    subprocess.run(subargs)


schedule.every().friday.at('02:30').do(
    train_model,
    config=('configs/timesformer/'
            'timesformer_divST_8x32x1_15e_kinetics400_rgb.py'),
    work_dir='mmaction2/work_dir/timesformer/')

while True:
    schedule.run_pending()
