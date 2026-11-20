import sys, time, runpy, torch
while True:
    free = torch.cuda.mem_get_info()[0] / 2**20
    if free >= 15000:
        break
    print(f'waiting for GPU memory, free={free:.0f} MiB', flush=True)
    time.sleep(30)
x = torch.empty(int(14 * 2**30), dtype=torch.uint8, device='cuda')
del x                      # stays reserved inside this process' PyTorch cache
print('reserved 14 GiB, starting training', flush=True)
sys.path[:0] = ['.', 'tools']
sys.argv = ['tools/train.py', '--config', 'configs/ft_shadow.py', '--experiment', 'full', '--seed', '1',
            '--data-root', 'data/cotton_709', '--work-dir', 'work_dirs/shadow_s1',
            '--load-from', 'work_dirs/full_s1/iter_10000.pth']
runpy.run_path('tools/train.py', run_name='__main__')
