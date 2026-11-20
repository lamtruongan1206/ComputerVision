base_scale = (
    2048,
    2048,
)
crop_size = (
    1024,
    1024,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'agri_pidnet',
        'shadow_aug',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/cotton_two_stage'
dataset_type = 'CottonFieldDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=1,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = 'work_dirs/full_s1/iter_10000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 10000
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        boundary_loss_weight=0.2,
        edge_width=4,
        head_planes=128,
        loss_decode=[
            dict(
                avg_non_ignore=True,
                class_weight=[
                    1.0,
                    1.0,
                    1.0,
                    5.0,
                ],
                loss_name='loss_ce',
                loss_weight=0.4,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                class_weight=[
                    0.5,
                    0.5,
                    0.5,
                    3.0,
                ],
                loss_name='loss_dice',
                loss_weight=0.25,
                type='AgriDiceLoss',
                use_sigmoid=False),
            dict(
                class_weight=[
                    1.0,
                    1.0,
                    1.0,
                    4.0,
                ],
                classes='all',
                loss_name='loss_lovasz',
                loss_type='multi_class',
                loss_weight=0.15,
                per_image=False,
                reduction='none',
                type='LovaszLoss'),
        ],
        m=2,
        n=3,
        num_classes=4,
        planes=32,
        ppm_planes=96,
        sampler=dict(min_kept=16384, thresh=0.7, type='OHEMPixelSampler'),
        type='AgriPIDNetSHead',
        use_boundary_loss=True,
        use_bwr_bag=True,
        use_ca=True,
        use_ca_mkir=True,
        use_mbcr=True,
        use_sae_d2t=True,
        use_window_attention=True),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='AgriEncoderDecoder')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0002, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=200, start_factor=0.001, type='LinearLR'),
    dict(
        begin=200,
        by_epoch=False,
        end=4000,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(deterministic=True, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root='data/cotton_two_stage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='CottonFieldDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=4000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root='data/cotton_two_stage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    1.5,
                ),
                scale=(
                    2048,
                    2048,
                ),
                type='RandomResize'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(prob=0.5, type='RandomShadow'),
            dict(type='PackSegInputs'),
        ],
        type='CottonFieldDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            1.5,
        ),
        scale=(
            2048,
            2048,
        ),
        type='RandomResize'),
    dict(crop_size=(
        1024,
        1024,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        data_root='data/cotton_two_stage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='CottonFieldDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/shadow_s1'
