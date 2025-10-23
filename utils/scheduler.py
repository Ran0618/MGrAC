import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, warmup_start_lr=0.0,
                                    base_lr=0.1, final_lr=0.01, num_cycles=1, last_epoch=-1):
    """
    Creates a schedule with a learning rate that decreases following
    the values of the cosine function between 0 and pi,
    after a warmup period during which it increases linearly from warmup_start_lr to base_lr.
    """

    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < num_warmup_steps:
            return warmup_start_lr + (base_lr - warmup_start_lr) * (
                    float(current_step) / float(max(1, num_warmup_steps)))

        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return final_lr + (base_lr - final_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(num_cycles * torch.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
