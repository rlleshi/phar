# * dataset settings
dataset_type = 'VideoDataset'
data_root = 'mmaction2/data/phar/'
data_root_val = data_root
data_root_test = data_root
ann_file_train = f'{data_root}/train_aug.txt'
ann_file_val = f'{data_root_val}/val.txt'
ann_file_test = f'{data_root_test}/val.txt'
num_classes = 17
img_norm_cfg = dict(mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_bgr=False)

# * model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=16,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.2,
        transformer_layers=None,
        # divided attention is the best strategy
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead',
                  num_classes=num_classes,
                  in_channels=768,
                  topk=(1, 2, 3, 4, 5)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

train_pipeline = [
    dict(type='DecordInit'),
    # * frame_interval has been selected for 7s clips
    dict(type='SampleFrames', clip_len=16, frame_interval=12, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=16,
         frame_interval=12,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=16,
         frame_interval=12,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(videos_per_gpu=1,
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
    interval=1,
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))),
)
# set the top-k accuracy during testing
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))), )

# optimizer
optimizer = dict(type='SGD',
                 lr=0.0015625,
                 momentum=0.9,
                 paramwise_cfg=dict(
                     custom_keys={
                         '.backbone.cls_token': dict(decay_mult=0.0),
                         '.backbone.pos_embed': dict(decay_mult=0.0),
                         '.backbone.time_embed': dict(decay_mult=0.0)
                     }),
                 weight_decay=1e-4,
                 nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 25

# * runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/timesformer/'
             'timesformer_divST_8x32x1_15e_kinetics400_rgb/'
             'timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth')
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
