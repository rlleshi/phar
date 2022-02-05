import logging
import os
import os.path as osp
import re
import subprocess
from argparse import ArgumentParser, ArgumentTypeError

import mlflow
from mmcv import Config, DictAction
from rich.console import Console

CONSOLE = Console()

# -----------------------------------------------------------------------------
# This script performs hyperparameter tuning for models based on Video dataset
# of MMaction2. It uses MLFlow Tracking to store its results
# ? OpTuna for hyperparameter tuning
#
# In a nutshell
# 1) Creates a dataset if it doesn't already exist
# 2) Trains a model on this dataset
# 3) Evaluates this model
#   3.1) Loss & Accuracy for the model (top1, top2, top3, top5)
#   3.2) Average training speed
#   3.3) Model complexity
#   3.4) Evaluate Accuracy Per Class
#   3.5) Create demo videos
#   3.6) Background influence check (via gradcam) on the two above videos
#   3.7) Test the model on the test dataset
# 4) Log everything with MLFlow
#
# See parse_args() for required arguments & defaults
# -----------------------------------------------------------------------------


def file_len(name):
    i = -1
    if not osp.exists(name):
        logging.error(f'file {name} does not exist')
    with open(name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def find_files(root, file_hint, ext):
    """Given a root folder, find a file based on part of its name and
    extension."""
    files = []
    for file in os.listdir(root):
        if (file_hint in file) & (osp.splitext(file)[1] == ext):
            files.append(file)
    return files
    # check = []
    # for root, _, files in os.walk(root):
    #     check += [file for file in files if file_hint in file]
    # if len(check) > 0:
    #     return [file for file in check if file.endswith(ext)]
    # return []


def train_test_parser(val):
    """must be between range 0 & 1."""
    try:
        val = float(val)
    except ValueError:
        raise ArgumentTypeError('{}: {} must be float'.format(val, type(val)))
    if val < 0.0 or val > 1.0:
        raise ArgumentTypeError('Test split out of bounds [0,1]')
    return val


def cleanup(args):
    for file in os.listdir(args.work_dir):
        path = osp.join(args.work_dir, file)
        try:
            if osp.isfile(path) or osp.islink(path):
                os.unlink(path)
        except Exception as e:
            logging.info(f'Could not delete {file}: {e}')


def try_delete_log_files():
    for log in find_files('/mmaction2/', 'tanz_', '.log'):
        os.unlink(osp.join('/mmaction2/', log))


def is_bast(args):
    """Check if a similar tanz dataset already exists based on its log file."""
    if args.cfg.dataset_type == 'PoseDataset':
        return True  # must have been generated beforehand

    def get_num_videos(dIr):
        return len([
            f for f in os.listdir(dIr)
            if ((f.endswith('mp4')) or (f.endswith('MTS')))
        ])

    tanz_log = 'tanz_' + args.ann_type + '_' + \
        str(get_num_videos(args.src_dir)) + 'vid_' + \
        str(args.test_split) + 'split_' + str(args.clip_length) + \
        's_clips_' + str(args.sliding_window) + 's_sliding_' + 'V1.log'
    return any(tanz_log == log
               for log in find_files('/mmaction2/', 'tanz_', '.log'))


def get_top_model(args):
    models = find_files(args.work_dir, 'best', '.pth')
    top_model = max(
        [int(re.findall(r'\d+',
                        model.split('_')[-1])[0]) for model in models])
    for model in models:
        if int(re.findall(r'\d+', model.split('_')[-1])[0]) == top_model:
            return model


def prettify(byte_content):
    decoded = byte_content.decode('utf-8')
    formatted_output = decoded.replace('\\n', '\n').replace('\\t', '\t')
    return formatted_output


def parse_args():
    default_config = ('/mmaction2/configs/recognition/i3d/'
                      'i3d_r50_video_32x2x1_bast_baseline.py')
    parser = ArgumentParser(prog='model tuning')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument(
        '--name', type=str, default='baseline', help='current run name')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='/mnt/data_transfer_tuning/write/work_dir',
        help='mlflow dir')
    parser.add_argument(
        '--config', type=str, default=default_config, help='model config file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--resume-from',
        action='store_true',
        help='if there is a ready model, start only the evaluation process')
    parser.add_argument(
        '--resume-from-training',
        type=str,
        help='the checkpoint file to resume the training from')
    parser.add_argument(
        '--execute-only',
        type=str,
        nargs='+',
        help='only the given functions will be executed')
    parser.add_argument(
        '--src-dir',
        type=str,
        default='/mnt/data_transfer/read/to_process/',
        help='raw videos directory')
    parser.add_argument(
        '--test-split',
        type=train_test_parser,
        default='0.18',
        help='dataset train/test ratio. Give for testing')
    parser.add_argument(
        '--clip-length', type=int, default=10, help='clip length for dataset')
    parser.add_argument(
        '--sliding-window',
        type=int,
        default=5,
        help='datasets clips sliding window')
    parser.add_argument(
        '--ann-type',
        default='base',
        choices=['base', 'eval'],
        help='type of annotations for which to generate the dataset')
    parser.add_argument(
        '--no-demo',
        action='store_true',
        help='whether to perform long video demos')
    args = parser.parse_args()
    return args


def log_artifacts_params(args):
    # log dataset params
    d_type = 'tanz_base' if args.ann_type == 'base' else 'tanz_evaluation'
    files_length = file_len(
        f'/mmaction2/data/{d_type}/tanz_train_list_videos.txt') + file_len(
            f'/mmaction2/data/{d_type}/tanz_val_list_videos.txt')
    validation_split = round(args.test_split * 83 / 98, 2)
    mlflow.log_params({
        'model':
        args.experiment.split('-')[0],
        'train_split':
        '~' + str(100 * (1 - validation_split) - 15) + '%',
        'validation_split':
        '~' + str(100 * (validation_split)) + '%',
        'test_split':
        '~15%',  # 15 out of 98 total videos, rest goes to train/val
        'clip_length':
        str(args.clip_length) + 's',
        'sliding_window':
        str(args.sliding_window) + 's',
        'number_of_clips':
        files_length,
    })

    # log training config & log files
    for ext in ['.json', '.log', '.py', '.png', '.jpg', '.csv', '.mp4']:
        for file in find_files(args.work_dir, '', ext):
            mlflow.log_artifact(osp.join(args.work_dir, file))

    # log top model
    top_model = get_top_model(args)
    if top_model is not None:
        mlflow.log_artifact(osp.join(args.work_dir, top_model))
    else:
        mlflow.log_param('top_model', '404 Not Found')

    # log dataset log file
    mlflow.log_artifact(find_files('/mmaction2/', 'tanz_', '.log')[0])


def generate_dataset(args):
    script_path = ('/mmaction2/human-action-recognition/har/tools'
                   '/data/BAST/generate_dataset_videos.py')
    subargs = [
        'python',
        script_path,
        args.src_dir,  # dir of raw videos
        '--test-split',
        str(args.test_split),
        '--clip-length',
        str(args.clip_length),
        '--sliding-window',
        str(args.sliding_window),
        '--ann-type',
        args.ann_type
    ]
    logging.info(subprocess.run(subargs))


def train(args):
    script_path = '/mmaction2/tools/dist_train.sh'
    no_gpus = 4
    subargs = [
        'bash', script_path, args.config,
        str(no_gpus), '--work-dir', args.work_dir, '--validate'
    ]
    if args.resume_from_training is not None:
        subargs.append('--resume-from')
        subargs.append(args.resume_from_training)
    if args.cfg_options:
        subargs.append('--cfg-options')
        for tup in args.cfg_options.items():
            subargs.append(f'{tup[0]}={tup[1]}')
    logging.info(subprocess.run(subargs))


def calc_loss_acc_training_speed(args):
    """Calculate the loss & accuracy during the model's training and it's
    training time."""

    script_path = '/mmaction2/tools/analysis/analyze_logs.py'
    json_log = find_files(args.work_dir, 'log', '.json')
    if len(json_log) == 0:
        logging.error(
            '# json log file for loss & accuracy calculations not found')
        return
    # if the training was interrupted, there can be more than 1 json logs
    json_log = [osp.join(args.work_dir, jl) for jl in json_log]
    config_file = find_files(args.work_dir, '', '.py')
    if len(config_file) > 0:
        name = config_file[0].split('_')
        title = name[0] + ' ' + osp.splitext(name[-1])[0]
    else:
        title = ''

    for i, log in enumerate(json_log):
        subargs_loss = [
            'python',
            script_path,
            'plot_curve',  # to plot curves
            log,  # training logs
            '--keys',
            'loss_cls',
            '--legend',
            'loss cls',
            '--title',
            f'{title}-{i}',
            '--style',
            'darkgrid',
            '--out',
            osp.join(args.work_dir, f'training_loss_{i}.png')
        ]
        result = subprocess.run(subargs_loss, capture_output=True)
        if prettify(result.stderr):
            # ! wrong error handling everywhere
            logging.error(f'Error with loss calculation: '
                          f'{prettify(result.stderr)}')

        subargs_acc = [
            'python',
            script_path,
            'plot_curve',  # to plot curves
            log,  # training logs
            '--keys',
            'top1_acc',
            'top2_acc',
            'top3_acc',
            '--legend',
            'top1 acc',
            'top2 acc',
            'top3 acc',
            '--title',
            f'{title}-{i}',
            '--style',
            'darkgrid',
            '--out',
            osp.join(args.work_dir, f'training_topk_acc_{i}.png')
        ]
        result = subprocess.run(subargs_acc, capture_output=True)
        if prettify(result.stderr):
            logging.error(f'Error with acc calculation: '
                          f'{prettify(result.stderr)}')

    result = subprocess.run(
        [
            'python',
            script_path,
            'cal_train_time',  # model train time
            json_log[0]
        ],
        capture_output=True)
    if prettify(result.stderr):
        logging.error('Error with training speed calculation: '
                      f'{prettify(result.stderr)}')
    else:
        mlflow.log_param('average_training_speed', prettify(result.stdout))
        mlflow.log_text(prettify(result.stdout), 'average_training_speed.txt')


def calc_model_complexity(args):
    """Calculate #of parameters & #flops for the model."""
    script_path = '/mmaction2/tools/analysis/get_flops.py'
    # for skeleton-based models
    # https://github.com/open-mmlab/mmaction2/issues/1022
    if args.cfg.dataset_type == 'PoseDataset':
        n_clips = 1
        clip_len = 48
        n_channels = 17
    else:
        n_clips = args.cfg.train_pipeline[1]['num_clips']
        clip_len = args.cfg.train_pipeline[1]['clip_len']
        n_channels = 3

    subargs = [
        'python',
        script_path,
        args.config,
        '--shape',
        str(n_clips),
        str(n_channels),  # channels
        str(256),  # height
        str(256),  # width
    ]

    if args.cfg.model.type == 'Recognizer3D':
        subargs.insert(6, str(clip_len))

    result = subprocess.run(subargs, capture_output=True)
    if prettify(result.stderr):
        logging.error('Error with model complexity calculation: '
                      f'{prettify(result.stderr)}')
    else:
        # number of flops is approximate
        mlflow.log_text(prettify(result.stdout), 'model_complexity.txt')


def eval_acc_per_class(args):
    """Accuracy per class calculated for each clip & not for each frame
    Training, Validation & Testing Clips."""
    script_path = ('/mmaction2/human-action-recognition/har/tools'
                   '/analysis/evaluate_acc_per_cls.py')
    top_model = get_top_model(args)
    if top_model is None:
        msg = '# Top Model Not Found (eval_acc_per_class)'
        mlflow.log_param('top_model', msg)
        logging.error(msg)
        return

    d_type = 'tanz_base' if args.ann_type == 'base' else 'tanz_evaluation'
    for split, split_dir in zip(['train', 'validation'],
                                ['videos_train', 'videos_val']):
        subargs = [
            'python',
            script_path,
            osp.join(args.work_dir, top_model),
            split,
            '--src-dir',
            osp.join(f'/mmaction2/data/{d_type}', split_dir),
            '--out',
            args.work_dir,
            '--config',
            args.config,
            '--labels',
            f'/mmaction2/data/{d_type}/annotations/tanz_annotations.txt',
            # '--device',
        ]
        result = subprocess.run(subargs, capture_output=True)
        if prettify(result.stderr):
            logging.error('Error with evaluating accuracy per class: '
                          f'{prettify(result.stderr)}')

    clips = 'clips' if args.ann_type == 'base' else 'clips_eval'
    subargs_test = [
        'python',
        script_path,
        osp.join(args.work_dir, top_model),
        'test',
        '--src-dir',
        f'/mnt/data_transfer/read/to_process_test/{clips}',
        '--out',
        args.work_dir,
        '--config',
        args.config,
        '--labels',
        f'/mmaction2/data/{d_type}/annotations/tanz_annotations.txt',
    ]
    result = subprocess.run(subargs_test, capture_output=True)
    if prettify(result.stderr):
        logging.error('Error with testing accuracy per class: '
                      f'{prettify(result.stderr)}')


def get_long_video_demo_clip(args, videos, top_model):
    script_path_demo_clips = ('human-action-recognition/har/tools/'
                              'long_video_demo_clips.py')
    if args.ann_type == 'base':
        annotations = ('human-action-recognition/har/annotations/BAST/base/'
                       'tanz_annotations.txt')
    else:
        annotations = ('human-action-recognition/har/annotations/BAST/eval/'
                       'tanz_annotations_42.txt')

    for vid in videos:
        demo_vid = 'demo_' + vid[0].split('/')[-1]
        subargs_demo = [
            'python',
            script_path_demo_clips,
            vid[0],  # video
            args.config,
            osp.join(args.work_dir, top_model),
            osp.join(args.work_dir, demo_vid),
            '--ann',  # general annotations
            annotations,
            '--type',
            'pose',
        ]
        if vid[1] is not None:
            subargs_demo += ['--ann-bast', vid[1]]
        result = subprocess.run(subargs_demo, capture_output=True)
        if prettify(result.stderr):
            logging.error(('Error with long video demo: '
                           f'{prettify(result.stderr)}'))


def get_long_video_demo_gradcam(args):
    script_path_demo_frame = '/mmaction2/demo/long_video_demo.py'
    script_path_gradcam = '/mmaction2/demo/demo_gradcam.py'
    test_video_path = '/mnt/data_transfer/read/to_process_test/'
    val_video_path = '/mnt/data_transfer/read/to_process/'
    top_model = get_top_model(args)
    if top_model is None:
        mlflow.log_param('top_model',
                         '# Top Model Not Found (eval_acc_per_class)')
        return

    d_type = 'tanz_base' if args.ann_type == 'base' else 'tanz_evaluation'
    videos = [
        (osp.join(test_video_path,
                  'j#03_1.mp4'), osp.join(test_video_path, 'V_3_std..eaf')),
        (osp.join(test_video_path,
                  '2019_04_1.mp4'), osp.join(test_video_path,
                                             '2019_04_1.eaf')),
        ((osp.join(val_video_path, '#107_1.mp4'), None))  # val set
    ]

    if args.ann_type == 'base':
        # for base annotations, we also want to see
        # if we can use them as categories
        videos.append((osp.join(test_video_path, 'avatar_vid',
                                '#000_2.mp4'), None))
        videos.append((osp.join(test_video_path, 'avatar_vid',
                                '2019_04_2.mp4'), None))
    else:
        # 2019_04_1 doesn't have eval ann ground truth annotations
        for v in [
            ('j#04_1.mp4', None),
            ('#003_1.mp4', None),
                # 'V_4_std.eaf''#003.eaf''#039.eaf''#108.eaf'
            ('#039_1.mp4', None),
            ('#108_1.mp4', None),
        ]:
            if v[1] is None:
                videos.append((osp.join(test_video_path, v[0]), v[1]))
            else:
                videos.append((osp.join(test_video_path,
                                        v[0]), osp.join(test_video_path,
                                                        v[1])))  # test

    if args.cfg.dataset_type == 'PoseDataset':
        get_long_video_demo_clip(args, videos, top_model)
        return

    for vid in videos:
        demo_vid = 'demo_' + vid[0].split('/')[-1]
        if d_type == 'tanz_base':
            threshold = str(0.1) if vid[0][-5] == '2' else str(0.2)
        else:
            threshold = str(0.08)

        subargs_demo = [
            'python',
            script_path_demo_frame,
            args.config,  # model config
            osp.join(args.work_dir, top_model),
            vid[0],
            f'/mmaction2/data/{d_type}/annotations/tanz_annotations.txt',
            osp.join(args.work_dir, demo_vid),
            '--threshold',
            threshold,  # prediction thr
            '--stride',
            str(0.05),  # sparse predictions
            '--label-color',
            str(240),
            str(230),
            str(0)
        ]
        result = subprocess.run(subargs_demo, capture_output=True)
        if prettify(result.stderr):
            logging.error(('Error with long video demo: '
                           f'{prettify(result.stderr)}'))

        subargs_gradcam = [
            'python',
            script_path_gradcam,
            args.config,
            osp.join(args.work_dir, top_model),
            osp.join(args.work_dir, demo_vid),
            '--out-filename',
            osp.join(args.work_dir, 'gradcam_demo_' + vid[0].split('/')[-1]),
            '--device',
            'cpu',
            '--fps',
            str(15),
            '--target-resolution',
            # str(480),
            str(800),
            str(854),
            '--target-layer-name',
            'backbone/layer4/2/relu'  # * adapt according to model
            # 'cls_head/fc_cls'
        ]
        result = subprocess.run(subargs_gradcam, capture_output=True)
        if prettify(result.stderr):
            logging.error(f'Error with gradcam: {prettify(result.stderr)}')


def test_model(args):
    script_path = '/mmaction2/tools/dist_test.sh'
    no_gpus = 4
    top_model = get_top_model(args)
    if top_model is None:
        msg = '# Top Model Not Found (eval_acc_per_class)'
        mlflow.log_param('top_model', msg)
        logging.error(msg)
        return

    subargs = [
        'bash',
        script_path,
        args.config,
        osp.join(args.work_dir, top_model),
        str(no_gpus),
        '--out',
        # * tools/analysis/eval_metric.py uses this
        osp.join(args.work_dir, 'test_results.json'),
        '--eval',
        'top_k_accuracy',
        'mean_class_accuracy'
    ]
    result = subprocess.run(subargs, capture_output=True)
    if prettify(result.stderr):
        logging.error(f'Error with model test: {prettify(result.stderr)}')

    result = prettify(result.stdout)
    result = result.split(
        'mnt/data_transfer_tuning/write/work_dir/test_results.json')[1]
    mlflow.log_text(result, 'test_results.txt')
    mlflow.log_param('test_results',
                     result.split('***')[0])  # Save only the final output


def main():
    args = parse_args()
    log_path = f'/mnt/data_transfer/write/{args.experiment}-logs.txt'
    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)
    args.cfg = Config.fromfile(args.config)

    if args.execute_only:
        if 'log_artifacts_params' in args.execute_only:
            mlflow.set_tracking_uri(args.work_dir)
            mlflow.set_experiment(args.experiment)
            mlflow.start_run(run_name=args.name)
        for func in args.execute_only:
            globals()[func](args)
        if 'log_artifacts_params' in args.execute_only:
            mlflow.end_run()
        return

    mlflow.set_tracking_uri(args.work_dir)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.name):
        if not args.resume_from:
            if not is_bast(args):
                try_delete_log_files()
                generate_dataset(args)
            train(args)

        calc_loss_acc_training_speed(args)
        calc_model_complexity(args)
        if args.cfg.dataset_type != 'PoseDataset':
            eval_acc_per_class(args)
        if not args.no_demo:
            get_long_video_demo_gradcam(args)
        test_model(args)
        log_artifacts_params(args)
        cleanup(args)


if __name__ == '__main__':
    main()
