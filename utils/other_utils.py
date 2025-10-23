import shutil
import numpy as np
import torch
import os


# 定义一个函数来进行动量更新
def momentum_update(source_model, target_model, momentum=0.99):
    for param_src, param_tgt in zip(source_model.parameters(), target_model.parameters()):
        param_tgt.data = momentum * param_tgt.data + (1.0 - momentum) * param_src.data


def save_checkpoint(state, is_best, save_path):
    filename = f'checkpoint.pth.tar'
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, f'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
