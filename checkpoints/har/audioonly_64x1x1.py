dataset_type = 'AudioFeatureDataset'
data_root = 'phar/mmaction2/data/phar/audio_feature/filtered_20/'
data_root_val = 'mmaction2/data/phar/audio_feature/filtered_20/'
data_root_test = 'mmaction2/data/phar/audio_feature/filtered_20/'
ann_file_train = 'mmaction2/data/phar/audio_feature/filtered_20//train.txt'
ann_file_val = 'mmaction2/data/phar/audio_feature/filtered_20//val.txt'
ann_file_test = 'mmaction2/data/phar/audio_feature/filtered_20//val.txt'
num_classes = 4
model = dict(type='AudioRecognizer',
             backbone=dict(type='ResNetAudio',
                           depth=101,
                           pretrained=None,
                           in_channels=1,
                           norm_eval=False),
             cls_head=dict(type='AudioTSNHead',
                           num_classes=4,
                           in_channels=1024,
                           dropout_ratio=0.5,
                           init_std=0.01),
             train_cfg=None,
             test_cfg=dict(average_clips='prob'))
train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames',
         clip_len=64,
         frame_interval=1,
         num_clips=1,
         test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames',
         clip_len=64,
         frame_interval=1,
         num_clips=10,
         test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(videos_per_gpu=16,
            workers_per_gpu=1,
            test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            train=dict(type='AudioFeatureDataset',
                       ann_file='train.txt',
                       data_prefix='',
                       pipeline=[
                           dict(type='LoadAudioFeature'),
                           dict(type='SampleFrames',
                                clip_len=64,
                                frame_interval=1,
                                num_clips=1),
                           dict(type='AudioFeatureSelector'),
                           dict(type='FormatAudioShape', input_format='NCTF'),
                           dict(type='Collect',
                                keys=['audios', 'label'],
                                meta_keys=[]),
                           dict(type='ToTensor', keys=['audios'])
                       ]),
            val=dict(type='AudioFeatureDataset',
                     ann_file='val.txt',
                     data_prefix='',
                     pipeline=[
                         dict(type='LoadAudioFeature'),
                         dict(type='SampleFrames',
                              clip_len=64,
                              frame_interval=1,
                              num_clips=1,
                              test_mode=True),
                         dict(type='AudioFeatureSelector'),
                         dict(type='FormatAudioShape', input_format='NCTF'),
                         dict(type='Collect',
                              keys=['audios', 'label'],
                              meta_keys=[]),
                         dict(type='ToTensor', keys=['audios'])
                     ]),
            test=dict(type='AudioFeatureDataset',
                      ann_file='val.txt',
                      data_prefix='',
                      pipeline=[
                          dict(type='LoadAudioFeature'),
                          dict(type='SampleFrames',
                               clip_len=64,
                               frame_interval=1,
                               num_clips=10,
                               test_mode=True),
                          dict(type='AudioFeatureSelector'),
                          dict(type='FormatAudioShape', input_format='NCTF'),
                          dict(type='Collect',
                               keys=['audios', 'label'],
                               meta_keys=[]),
                          dict(type='ToTensor', keys=['audios'])
                      ]))
evaluation = dict(interval=5,
                  metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                5))))
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))))
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 240
checkpoint_config = dict(interval=20)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = 'mmaction2/work_dir/audio/'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
