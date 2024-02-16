import os
import warnings
warnings.simplefilter("ignore") 

custom_imports = dict(imports=["geospatial_fm"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True

dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "/home/ziggy/devel/kelp_data/"


num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4

img_norm_cfg = dict(
    means = [10035.471588303459, 10735.157409510864, 8749.484468334336, 8870.374927721672, 8555.930040329898, 0.07059119714611709, 14.856754975621637],
    # means=[
        
    #     # 0.2323245113436119,
    #     # 0.11944914225186566,
    #     # 0.05889748132001316,
    #     # 0.05701185520536176,
    #     # 0.033349706741586264,
    #     # 0.1972854853760658,
    # ],
    stds= [2804.9866899590766, 3392.3579149885463, 1430.6976191883807, 1302.9422392186511, 1168.9829945994245, 0.16116380862181137, 23.389279097947544],
    # stds=[
        
    #     # 0.08708738838140137,
    #     # 0.07241979477437814,
    #     # 0.04004109844362779,
    #     # 0.026807560223070237,
    #     # 0.02269135568823774,
    #     # 0.07791732423672691,
    #    ],
)  # change the mean and std of all the bands

bands = [0, 1, 2, 3, 4, 5, 6]
tile_size = 224
orig_nsize = 350
crop_size = (tile_size, tile_size)
img_suffix = "_satellite.tif"
seg_map_suffix = "_kelp.tif"
ignore_index = -1
image_nodata = -32_768
image_nodata_replace = 0
image_to_float32 = True

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "/home/ziggy/devel/Prithvi-100M/Prithvi_100M.pt"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = num_frames * embed_dim
max_intervals = 20000
evaluation_interval = 500

# TO BE DEFINED BY USER: model path
experiment = "prithvi-retry-02-16"
project_dir = "kelp-me"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

save_path = work_dir
train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=image_to_float32),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=False),
    dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=(tile_size, tile_size)),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(num_frames,len(bands), tile_size, tile_size),
    ),
    dict(type="TorchGaussianBlur",kernel_size=5, sigma=(0.1, 3)),
    dict(type="TorchColorJitter", brightness=(0.8,1.2), contrast=(0.8,1.2)),
    dict(type="TorchRandomAffine", degrees=(-180, 180), translate=(0.15, 0.15), scale=(0.7, 1), shear=(-10, 10,-10,10)),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=image_to_float32),
    dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up=dict({"2": 1, "3": 2}),
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

CLASSES = ("Not Kelp", "Kelp")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="train_satellite",
        ann_dir="train_kelp",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=-1,
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="val_satellite",
        ann_dir="val_kelp",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=-1,
    ),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="test_satellite",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=-1,
    ),
)

optimizer = dict(type="Adam", lr=1.3e-05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook", by_epoch=False),
        dict(type='MMSegWandbHook', by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
             init_kwargs=dict(project=project_dir, name=experiment),
             interval=500,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=10)

    ],
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project=project_dir, name=experiment))]

visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)


checkpoint_config = dict(by_epoch=False, interval=500, out_dir=save_path)
# save the best checkpoint by dice
evaluation = dict(
    interval=evaluation_interval,
    metric="mDice",
    pre_eval=True,
    by_epoch=False,
)

loss_func = dict(type="DiceLoss", use_sigmoid=False, loss_weight=1, ignore_index=-1)

runner = dict(type="IterBasedRunner", max_iters=max_intervals)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
auto_resume = False
