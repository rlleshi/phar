# * dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/rejnald/projects/side_projects/phar/mmaction2/data/phar/'
data_root_val = data_root
data_root_test = data_root
ann_file_train = f'{data_root}/train_aug.txt'
ann_file_val = f'{data_root_val}/val.txt'
ann_file_test = f'{data_root_test}/val.txt'
num_classes = 17

# * model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet3dSlowOnly',
                  depth=50,
                  pretrained='torchvision://resnet50',
                  lateral=False,
                  conv1_kernel=(1, 7, 7),
                  conv1_stride_t=1,
                  pool1_stride_t=1,
                  inflate=(0, 0, 1, 1),
                  norm_eval=False,
                  non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0),
                             (0, 0, 0)),
                  non_local_cfg=dict(sub_sample=True,
                                     use_scale=True,
                                     norm_cfg=dict(type='BN3d',
                                                   requires_grad=True),
                                     mode='embedded_gaussian')),
    cls_head=dict(type='I3DHead',
                  in_channels=2048,
                  num_classes=num_classes,
                  spatial_type='avg',
                  dropout_ratio=0.7,
                  topk=(1, 2, 3, 4, 5)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

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
    dict(type='CenterCrop', crop_size=224),
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
data = dict(videos_per_gpu=4,
            workers_per_gpu=1,
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

# * optimizer
optimizer = dict(type='SGD', lr=0.00625, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 step=[90, 130],
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=10)
total_epochs = 150

# * runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'
             'slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/'
             'slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_'
             '20210308-e8dd9e82.pth')
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
