_base_ = './agri_pidnet_s.py'
max_iters = 10000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=750),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, by_epoch=False, begin=750, end=max_iters),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=1, save_last=True))
train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
resume = True
