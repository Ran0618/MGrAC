import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import time
from tqdm import tqdm
from .other_utils import AverageMeter

'''
直接排除预测为已知类的样本，并判定为错误分类实例
在预测为新类的样本和目标之间执行匈牙利算法,计算最优匹配
'''


@torch.no_grad()
def hungarian_evaluate(predictions, targets, offset=0):
    # Hungarian matching
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)

    # only consider the valid predicts. rest are treated as misclassification
    valid_idx = np.where(predictions_np >= 0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = max(torch.unique(predictions_sel).numel(), num_classes)

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred,
                             targets_k=num_classes)  # match is data dependent
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    # Gather performance metrics
    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(
        num_elems)  # accuracy is normalized with the total number of samples not only the valid ones
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())

    return {'acc': acc * 100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


# ORCA Cluster Evaluate
def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test_seen(args, test_loader, model, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()

    test_loader = tqdm(test_loader, desc="Testing seen:")

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            targets = targets.to(torch.long)
            outputs, _, _ = model(inputs)
            pred1, pred5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(pred1.item(), inputs.shape[0])
            top5.update(pred5.item(), inputs.shape[0])
            test_loader.set_postfix({"epoch": epoch+1, "top1": top1.avg, "top5": top5.avg})
        test_loader.close()

    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0, test_type=None):
    gt_targets = []
    predictions = []
    model.eval()

    if args.show_progress:
        test_loader = tqdm(test_loader, desc=f"Testing {test_type}")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs, _, _ = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
        if args.show_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)

    return eval_output
