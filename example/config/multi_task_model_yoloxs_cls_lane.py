_base_ = [
    '../../third_party/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '../../third_party/mmdetection/configs/_base_/default_runtime.py',
    '../../third_party/mmdetection/configs/yolox/yolox_tta.py'
]

custom_imports = dict(
    imports=[
        "datasets",
        "evaluation",
        "models",
        "example"
    ],
    allow_failed_imports=False,
)

TASK_CLASSES_MAP = {
    "classification": {
        "Residential area houses": 0,
        "Urban": 1,
        "Suburbs": 2,
        "Highway": 3,
        "Parking lot / Exits": 4,
        "Clear": 5,
        "Partly Cloudy": 6,
        "Overcast": 7,
        "Rainy": 8,
        "Foggy": 9,
        "Daytime": 10,
        "Night": 11,
        "Dawn/Dusk": 12
    },
    "scene_classification": {
        "Residential area houses": 0,
        "Urban": 1,
        "Suburbs": 2,
        "Highway": 3,
        "Parking lot / Exits": 4
    },
    "weather_classification": {
        "Clear": 0,
        "Partly Cloudy": 1,
        "Overcast": 2,
        "Rainy": 3,
        "Foggy": 4
    },
    "hours_classification": {
        "Daytime": 0,
        "Night": 1,
        "Dawn/Dusk": 2
    }
}

METAINFO = {
    "bbox_classes": ["TYPE_UNKNOWN", "TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_SIGN", "TYPE_CYCLIST"],
    "lane_kpts_classes": ["left-left", "left", "right", "right-right"],
    "classification_classes": [
        "Residential area houses",
        "Urban",
        "Suburbs",
        "Highway",
        "Parking lot / Exits",
        "Clear",
        "Partly Cloudy",
        "Overcast",
        "Rainy",
        "Foggy",
        "Daytime",
        "Night",
        "Dawn/Dusk"
    ],
    "scene_classification_classes": ["Residential area houses", "Urban", "Suburbs", "Highway", "Parking lot / Exits"],
    "weather_classification_classes": ["Clear", "Partly Cloudy", "Overcast", "Rainy", "Foggy"],
    "hours_classification_classes": ["Daytime", "Night", "Dawn/Dusk"],
}
# 864 x 576
# 768 x 512
# 672 x 448
# 576 x 384
# 480 x 320
RESIZE_IMG = (480, 320) # width, height (maintaining aspect ratio of 3:2)
MODEL_INPUT = (480, 320)  # width, height (after croping)

# dataset settings
data_root = "/mnt/e/WORK/DATA/OpenLane/example_openlane_data/"
dataset_type = 'OpenLaneDataset'

IMG_PREFIX = data_root + 'images/'
ANN_PREFIX = data_root + 'annotations/'

USE_CACHED_DATA=False

# model settings
model = dict(
    type='MultiTaskModel',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    heads=dict(
        bbox_head=dict(
            arch=dict(
                type='YOLOXHead',
                num_classes=80,
                in_channels=128,
                feat_channels=128,
                stacked_convs=2,
                strides=(8, 16, 32),
                use_depthwise=False,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish'),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='IoULoss',
                    mode='square',
                    eps=1e-16,
                    reduction='sum',
                    loss_weight=5.0),
                loss_obj=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0),
                loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5)),
                # In order to align the source code, the threshold of the val phase is
                # 0.01, and the threshold of the test phase is 0.001.
                test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
            ),
            in_variables=['neck_x'],
            l_kwargs={},
            p_kwargs={'rescale': True},
            predict_mode=True,
            task_type="object_detection"
        ),
        lane_kpts_head=dict(
            arch=dict(
                type='UltraFastLaneDetectionV2',
                in_feature_size=(32, 20, 40),
                feat_idx=0,
                num_anchors_row=32,
                num_bins_row=64, 
                num_lanes_on_row=4, 
                num_anchors_col=0, 
                num_bins_col=0,
                num_lanes_on_col=0,
                lanes_from_anchors_row=[0,1,2,3],
                lanes_from_anchors_col=[],
                mlp_mid_dim=64,
            ),
            in_variables=['neck_x'],
            p_kwargs={'rescale': True},
            predict_mode=True,
            task_type="lane_kpts_detection"
        ),
        classification_head=dict(
            arch=dict(
                type='ClassificationHead',
                in_channels=512, # 1024 * widen_factor
                classes=len(METAINFO['classification_classes']),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=10)
            ),
            in_variables=['back_x'],
            kwargs={},
            predict_mode=True,
            task_type='classification',
        )
    ),
    train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)

backend_args = None

train_pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(type='MultiTaskRandomFlip', prob=0.5),
    dict(type='MultiTaskResize', scale=RESIZE_IMG, keep_ratio=True),
    dict(type='FixedCrop', crop_tl=(0,0), crop_br=MODEL_INPUT, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='mmdet.PackMultiTaskInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        image_set_file = None,
        ann_prefix = ANN_PREFIX,
        img_prefix = IMG_PREFIX,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(
                type='LoadMultiTaskAnnotations', 
                with_bbox=True,
                with_lane_kpts=True,
                with_classification_labels=True,
            )
        ],
        task_classes_map=TASK_CLASSES_MAP,
        metainfo=METAINFO,
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        use_cache=USE_CACHED_DATA,
        backend_args=backend_args),
    pipeline=train_pipeline)

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='MultiTaskResize', scale=RESIZE_IMG, keep_ratio=True),
    dict(type='FixedCrop', crop_tl=(0,0), crop_br=MODEL_INPUT, allow_negative_crop=True),
    dict(type='LoadMultiTaskAnnotations', with_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='mmdet.PackMultiTaskInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='MultiTaskResize', scale=RESIZE_IMG, keep_ratio=True),
    dict(type='FixedCrop', crop_tl=(0,0), crop_br=MODEL_INPUT, allow_negative_crop=True),
    dict(
        type='mmdet.PackMultiTaskInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=256,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=False,
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=512,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        image_set_file = None,
        ann_prefix = ANN_PREFIX,
        img_prefix = IMG_PREFIX,
        test_mode=True,
        pipeline=val_pipeline,
        task_classes_map=TASK_CLASSES_MAP,
        metainfo=METAINFO,
        use_cache=USE_CACHED_DATA,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=512,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        image_set_file = None,
        ann_prefix = ANN_PREFIX,
        img_prefix = IMG_PREFIX,
        test_mode=True,
        pipeline=test_pipeline,
        task_classes_map=TASK_CLASSES_MAP,
        metainfo=METAINFO,
        use_cache=USE_CACHED_DATA,
        backend_args=backend_args))

val_evaluator = dict(
    type='mmdet.DetectionMetric',
    metric='bbox',
    prefix="object_detection",
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator

# training settings
max_epochs = 300
num_last_epochs = 25
interval = 1

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.02
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3,  # only keep latest 3 checkpoints
        save_best=["detection/bbox_mAP"],
        rule="greater",
    )
)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

#_base_.visualizer.type = "MultiTaskVisualizer"
_base_.visualizer.vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
