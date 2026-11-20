from mmengine.config import Config
cfg = Config.fromfile('configs/exp.py')
ds = cfg.train_dataloader.dataset
while 'dataset' in ds:
    ds = ds['dataset']
pipe = ds['pipeline']
print('old pipeline:', [p.get('type') for p in pipe])
i = [p.get('type') for p in pipe].index('PackSegInputs')
pipe.insert(i, dict(type='PhotoMetricDistortion'))
pipe.insert(i + 1, dict(type='RandomShadow', prob=0.5))
print('new pipeline:', [p.get('type') for p in pipe])
old = cfg.get('custom_imports', None)
imports = list(old['imports']) if old else []
cfg.custom_imports = dict(imports=imports + ['shadow_aug'], allow_failed_imports=False)
cfg.load_from = 'work_dirs/full_s1/iter_10000.pth'
cfg.resume = False                      # resume=True + load_from would restart at iter 10000
cfg.train_cfg.max_iters = 4000
cfg.train_cfg.val_interval = 1000
cfg.optim_wrapper.optimizer.lr = 2e-4
for s in cfg.param_scheduler:
    if s['type'] == 'LinearLR':
        s['begin'], s['end'] = 0, 200
    else:
        s['begin'], s['end'] = 200, 4000
cfg.default_hooks.checkpoint.interval = 1000
cfg.work_dir = 'work_dirs/shadow_s1'
cfg.dump('configs/ft_shadow.py')
print('custom_imports:', cfg.custom_imports)
print('schedulers:', [(s['type'], s.get('begin'), s.get('end')) for s in cfg.param_scheduler])
print('lr', cfg.optim_wrapper.optimizer.lr, 'batch', cfg.train_dataloader.batch_size, 'saved configs/ft_shadow.py')
