# * dataset settings
dataset_type = 'AudioFeatureDataset'
data_root = ('/home/rejnald/projects/side_projects/phar/mmaction2/data/phar/'
             'audio_feature/filtered_20/')
data_root_val = data_root
data_root_test = data_root
ann_file_train = f'{data_root}/train.txt'
ann_file_val = f'{data_root_val}/val.txt'
ann_file_test = f'{data_root_test}/test.txt'
num_classes = 3

# * model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(type='AudioTSNHead',
                  num_classes=num_classes,
                  in_channels=512,
                  dropout_ratio=0.7,
                  init_std=0.01,
                  topk=(1, 2, 3, 4, 5)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=64, frame_interval=2, num_clips=1),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames',
         clip_len=64,
         frame_interval=2,
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
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(videos_per_gpu=32,
            workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            train=dict(type=dataset_type,
                       ann_file=ann_file_train,
                       data_prefix='',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     ann_file=ann_file_val,
                     data_prefix='',
                     pipeline=val_pipeline),
            test=dict(type=dataset_type,
                      ann_file=ann_file_test,
                      data_prefix='',
                      pipeline=test_pipeline))
# set the top-k accuracy during validation
evaluation = dict(
    interval=5,  # Interval to perform evaluation
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))),
)
# set the top-k accuracy during testing
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))), )

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 200

# * runtime settings
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = (
    'https://download.openmmlab.com/mmaction/recognition/'
    'audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/'
    'tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth')
# load_from=None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
