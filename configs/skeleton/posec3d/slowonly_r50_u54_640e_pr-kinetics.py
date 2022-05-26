# * dataset settings
dataset_type = 'PoseDataset'
data_root = ('/home/jovyan/mmaction2/data')
data_root_val = data_root
data_root_test = data_root
ann_file_train = f'{data_root}/kinesphere_train.pkl'
ann_file_val = f'{data_root_val}/kinesphere_val.pkl'
ann_file_test = f'{data_root_test}/kinesphere_val.pkl'
num_classes = 6
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# model settings
model = dict(type='Recognizer3D',
             backbone=dict(type='ResNet3dSlowOnly',
                           depth=50,
                           pretrained=None,
                           in_channels=17,
                           base_channels=32,
                           num_stages=3,
                           out_indices=(2, ),
                           stage_blocks=(4, 6, 3),
                           conv1_stride_s=1,
                           pool1_stride_s=1,
                           inflate=(0, 1, 1),
                           spatial_strides=(2, 2, 2),
                           temporal_strides=(1, 1, 2),
                           dilations=(1, 1, 1)),
             cls_head=dict(type='I3DHead',
                           in_channels=512,
                           num_classes=num_classes,
                           spatial_type='avg',
                           dropout_ratio=0.7,
                           topk=(1, 2, 3, 4, 5)),
             train_cfg=dict(),
             test_cfg=dict(average_clips='prob'))

train_pipeline = [
    # * 54 (25% of 210) sampled frames seems better
    # 48 frames = 22.8%
    dict(type='UniformSampleFrames', clip_len=54),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=54, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=54, num_clips=10,
         test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False,
         double=True,
         left_kp=left_kp,
         right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(videos_per_gpu=12,
            workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1),
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

# optimizer
optimizer = dict(type='SGD', lr=0.0375, momentum=0.9,
                 weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 640
checkpoint_config = dict(interval=40)
workflow = [('train', 10)]
evaluation = dict(interval=5,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1, 2, 3, 4, 5))
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))), )
log_config = dict(interval=20, hooks=[
    dict(type='TextLoggerHook'),
])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = (
    'https://download.openmmlab.com/mmaction/skeleton/posec3d/'
    'slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint/'
    'slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint'
    '-cae8aa4a.pth')
resume_from = None
find_unused_parameters = False
