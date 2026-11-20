_base_ = './agri_pidnet_s.py'
max_iters = 300
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=max_iters)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, by_epoch=False, begin=30, end=max_iters),
]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=max_iters))
train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
