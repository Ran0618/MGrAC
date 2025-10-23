import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .other_utils import AverageMeter


def uncertain_generator(args, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_max_std = []
    model.eval()

    data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (
                inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, targets,
                indexs) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs1 = inputs1.cuda(non_blocking=True)
            inputs2 = inputs2.cuda(non_blocking=True)
            inputs3 = inputs3.cuda(non_blocking=True)
            inputs4 = inputs4.cuda(non_blocking=True)
            inputs5 = inputs5.cuda(non_blocking=True)
            inputs6 = inputs6.cuda(non_blocking=True)
            inputs7 = inputs7.cuda(non_blocking=True)
            inputs8 = inputs8.cuda(non_blocking=True)
            inputs9 = inputs9.cuda(non_blocking=True)
            inputs10 = inputs10.cuda(non_blocking=True)
            out_prob = []

            outputs, _, _ = model(inputs1)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs2)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs3)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs4)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs5)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs6)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs7)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs8)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs9)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs, _, _ = model(inputs10)
            out_prob.append(F.softmax(outputs, dim=1))

            # compute uncertainty scores
            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob, dim=0)
            out_prob = torch.mean(out_prob, dim=0)
            _, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1, 1))

            pseudo_max_std.extend(max_std.squeeze(1).cpu().numpy().tolist())
            pseudo_idx.extend(indexs.numpy().tolist())

            batch_time.update(time.time() - end)
            end = time.time()
            data_loader.set_description("UncertainGen Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s.".format(
                batch=batch_idx + 1,
                iter=len(data_loader),
                data=data_time.avg,
                bt=batch_time.avg,
            ))

        data_loader.close()

    pseudo_max_std = np.array(pseudo_max_std)
    pseudo_idx = np.array(pseudo_idx)

    # normalizing the uncertainty values
    pseudo_max_std = pseudo_max_std / max(pseudo_max_std)
    pseudo_max_std = np.clip(pseudo_max_std, args.temperature, 1.0)

    uncertain_temp = {'index': pseudo_idx.tolist(), 'uncertain': pseudo_max_std.tolist()}

    return uncertain_temp
