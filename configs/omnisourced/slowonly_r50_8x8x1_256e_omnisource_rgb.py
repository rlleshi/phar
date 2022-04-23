# * dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/rejnald/projects/side_projects/phar/mmaction2/data/phar/'
data_root_val = data_root
data_root_test = data_root
ann_file_train = f'{data_root}/train.txt'
ann_file_val = f'{data_root_val}/val.txt'
ann_file_test = f'{data_root_test}/test.txt'
num_classes = 18

# * model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet3dSlowOnly',
                  depth=50,
                  pretrained=None,
                  lateral=False,
                  conv1_kernel=(1, 7, 7),
                  conv1_stride_t=1,
                  pool1_stride_t=1,
                  inflate=(0, 0, 1, 1),
                  norm_eval=False),
    cls_head=dict(type='I3DHead',
                  in_channels=2048,
                  num_classes=num_classes,
                  spatial_type='avg',
                  dropout_ratio=0.5,
                  topk=(1, 2, 3, 4, 5)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# * dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=8,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=8,
         num_clips=10,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(videos_per_gpu=6,
            workers_per_gpu=1,
            test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            train=dict(
                type=dataset_type,
                ann_file=ann_file_train,
                data_prefix='',
                pipeline=train_pipeline,
            ),
            val=dict(
                type=dataset_type,
                ann_file=ann_file_val,
                data_prefix='',
                pipeline=val_pipeline,
            ),
            test=dict(
                type=dataset_type,
                ann_file=ann_file_test,
                data_prefix='',
                pipeline=test_pipeline,
            ))
# set the top-k accuracy during validation
evaluation = dict(
    interval=5,  # Interval to perform evaluation
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))),
)
# set the top-k accuracy during testing
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))), )

# * optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)

# The flag indicates using joint training
# omnisource = True

# * runtime settings
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
total_epochs = 256
find_unused_parameters = False
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = (
    'https://download.openmmlab.com/mmaction/recognition/slowonly'
    '/omni/slowonly_r50_omni_4x16x1_kinetics400_rgb_20200926-51b1f7ea.pth')
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
