dataset_type = 'VideoDataset'
data_root = 'phar/mmaction2/data/phar/'
data_root_val = 'phar/mmaction2/data/phar/'
data_root_test = 'phar/mmaction2/data/phar/'
ann_file_train = 'phar/mmaction2/data/phar//train_aug.txt'
ann_file_val = 'phar/mmaction2/data/phar//val.txt'
ann_file_test = 'phar/mmaction2/data/phar//val.txt'
num_classes = 17
img_norm_cfg = dict(mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_bgr=False)
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=('https://download.openmmlab.com/mmaction/recognition/'
                    'timesformer/vit_base_patch16_224.pth'),
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.2,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(type='TimeSformerHead',
                  num_classes=17,
                  in_channels=768,
                  topk=(1, 2, 3, 4, 5)),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=24, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize',
         mean=[127.5, 127.5, 127.5],
         std=[127.5, 127.5, 127.5],
         to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=24,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize',
         mean=[127.5, 127.5, 127.5],
         std=[127.5, 127.5, 127.5],
         to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=24,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize',
         mean=[127.5, 127.5, 127.5],
         std=[127.5, 127.5, 127.5],
         to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(videos_per_gpu=1,
            workers_per_gpu=1,
            test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
            train=dict(type='VideoDataset',
                       ann_file='phar/mmaction2/data/phar//train_aug.txt',
                       data_prefix='',
                       pipeline=[
                           dict(type='DecordInit'),
                           dict(type='SampleFrames',
                                clip_len=8,
                                frame_interval=24,
                                num_clips=1),
                           dict(type='DecordDecode'),
                           dict(type='RandomRescale', scale_range=(256, 320)),
                           dict(type='RandomCrop', size=224),
                           dict(type='Flip', flip_ratio=0.5),
                           dict(type='Normalize',
                                mean=[127.5, 127.5, 127.5],
                                std=[127.5, 127.5, 127.5],
                                to_bgr=False),
                           dict(type='FormatShape', input_format='NCTHW'),
                           dict(type='Collect',
                                keys=['imgs', 'label'],
                                meta_keys=[]),
                           dict(type='ToTensor', keys=['imgs', 'label'])
                       ]),
            val=dict(type='VideoDataset',
                     ann_file='phar/mmaction2/data/phar//val.txt',
                     data_prefix='',
                     pipeline=[
                         dict(type='DecordInit'),
                         dict(type='SampleFrames',
                              clip_len=8,
                              frame_interval=24,
                              num_clips=1,
                              test_mode=True),
                         dict(type='DecordDecode'),
                         dict(type='Resize', scale=(-1, 256)),
                         dict(type='CenterCrop', crop_size=224),
                         dict(type='Normalize',
                              mean=[127.5, 127.5, 127.5],
                              std=[127.5, 127.5, 127.5],
                              to_bgr=False),
                         dict(type='FormatShape', input_format='NCTHW'),
                         dict(type='Collect',
                              keys=['imgs', 'label'],
                              meta_keys=[]),
                         dict(type='ToTensor', keys=['imgs', 'label'])
                     ]),
            test=dict(type='VideoDataset',
                      ann_file='phar/mmaction2/data/phar//val.txt',
                      data_prefix='',
                      pipeline=[
                          dict(type='DecordInit'),
                          dict(type='SampleFrames',
                               clip_len=8,
                               frame_interval=24,
                               num_clips=1,
                               test_mode=True),
                          dict(type='DecordDecode'),
                          dict(type='Resize', scale=(-1, 224)),
                          dict(type='ThreeCrop', crop_size=224),
                          dict(type='Normalize',
                               mean=[127.5, 127.5, 127.5],
                               std=[127.5, 127.5, 127.5],
                               to_bgr=False),
                          dict(type='FormatShape', input_format='NCTHW'),
                          dict(type='Collect',
                               keys=['imgs', 'label'],
                               meta_keys=[]),
                          dict(type='ToTensor', keys=['imgs', 'label'])
                      ]))
evaluation = dict(interval=1,
                  metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                5))))
eval_config = dict(metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4,
                                                                 5))))
optimizer = dict(
    type='SGD',
    lr=0.0015625,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        })),
    weight_decay=0.0001,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(interval=1000, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/timesformer/'
             'timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_'
             '8x32x1_15e_kinetics400_rgb-3f8e5d03.pth')
resume_from = 'mmaction2/work_dir/timesformer/latest.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = 'mmaction2/work_dir/timesformer/'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
