import torch


# logits is similarity matrix, mask is labels matrix, diagonal_mask is diagonal matrix
#     Differences between SupCon：
#     1. temperature
#     2. minus max_logit, value stable
def wcl_contra(logits, temperature, labels):
    # 相似度除以温度
    logits = logits / temperature
    # 找出每行的最大值，并确保每行减去最大值以增强数值稳定性
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    # 根据标签生成掩码矩阵
    mask = generate_mask_matrix(labels)
    # 去除对角线上的自相似度
    diagonal_mask = torch.eye(logits.shape[0]).cuda(non_blocking=True)
    if diagonal_mask is not None:
        diagonal_mask = 1 - diagonal_mask
        mask = mask * diagonal_mask
        exp_logits = torch.exp(logits) * diagonal_mask
    else:
        exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = (-mean_log_prob_pos).mean()

    return loss


def generate_mask_matrix(labels):
    # convert to tensor
    # labels = torch.tensor(labels, dtype=torch.long)

    # generate mask matrix, same class is 1, other is 0.
    matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).long()

    return matrix
