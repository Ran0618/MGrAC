import argparse
import itertools
import os
import time
from datetime import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from data import datasets
from networks.build import build_model
from objective.supcon_loss import SupConLoss
from utils.ema_optimizer import WeightEMA
from utils.evaluate_utils import test_seen, test_cluster
from utils.interleave import interleave
from utils.other_utils import AverageMeter, save_checkpoint
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.uncertain_eval import uncertain_generator
from tensorboardX import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description='GraCon Training')

    # 数据参数 #
    parser.add_argument('--data_root', default='../datasets', metavar='./root', type=str,
                        help='directory to store data')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars',
                                 'imagenet100', "cifar100_20"], help='dataset name')
    parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
    parser.add_argument('--out', default='outputs/', type=str, help='save root directory')

    # 优化参数 #
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument('--train_iteration', type=int, default=1024, help='number of iteration per epoch')
    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # 分类器学习率设置：0.005 * sqrt(batch-size)
    parser.add_argument("--base_lr", default=0.3, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.03, help="final learning rate")
    parser.add_argument("--warmup_start_lr", default=0.03, type=float, help="initial warmup learning rate")
    # 对比学习学习率设置：0.1 * sqrt(batch-size)
    parser.add_argument("--contrastive_lr", type=float, default=1, help="final learning rate")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument('--ema_decay', default=0.999, type=float, help="ema model weight decay")
    parser.add_argument('--use_cuda', default=True, type=bool, help="whether use CUDA")
    parser.add_argument('--show_progress', default=True, type=bool, help="whether show testing progress bar")

    # 开放世界半监督学习参数 #
    parser.add_argument('--labeled_percent', default=10, type=int, metavar='N', help='percentage of labeled data')
    parser.add_argument('--novel_percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--imbalance_factor', default=1, type=int, metavar='N', help='imbalance factor of the data')

    # GraCon参数 #
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50'],
                        help='model architecture')
    parser.add_argument('--num_fine', default=200, type=int, metavar='N', help='number of fine-grained')
    parser.add_argument('--num_coarse', default=20, type=int, metavar='N', help='number of coarse-grained')
    parser.add_argument('--hidden_mlp', default=512, type=int, metavar='N', help='hidden layer of projection head')
    parser.add_argument('--feat_dim', default=128, type=int, metavar='N', help='output dimension of projection head')
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--uncertain_freq", default=1, type=int, help="frequency of generating uncertainty scores")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for novel class hard pseudo-labeling")
    parser.add_argument('--alpha', default=0.75, type=float, help='mixup degree parameter')
    parser.add_argument('--mode', default='v2', type=str, help='use which version model')

    # 指定随机种子
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

    args = parser.parse_args()

    # 记录最优准确率
    best_acc = 0.0
    # 获取当前时间
    run_started = datetime.today().strftime('%d-%m-%y_%H%M%S')
    # 设置保存文件的位置
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_lbl_percent_{args.labeled_percent}' \
                    f'_novel_percent_{args.novel_percent}_{run_started}'
    args.out = os.path.join(args.out, args.exp_name)

    os.makedirs(args.out, exist_ok=True)

    # 确定已知类和新类数量
    if args.dataset == "cifar10":
        args.no_class = 10
    if args.dataset == "cifar100":
        args.no_class = 100
    args.no_seen = args.no_class - int((args.novel_percent * args.no_class) / 100)

    # 记录参数
    with open(f'{args.out}/parameters.txt', 'a+') as out_file:
        for k, v in vars(args).items():
            out_file.write(f'{k}={v}\n')

    # 设置随机数种子
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)

    # 加载数据集
    my_data = datasets.get_dataset_class(args)

    train_labeled_dataset, train_unlabeled_dataset, uncertain_dataset, \
        test_dataset_all, test_dataset_seen, test_dataset_novel = my_data.get_dataset()

    # 获取Dataloader
    labeled_train_loader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, drop_last=True, pin_memory=True)
    unlabeled_train_loader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True, pin_memory=True)
    uncertain_loader = data.DataLoader(uncertain_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)
    test_loader_all = data.DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False,
                                      num_workers=0, pin_memory=True)
    test_loader_seen = data.DataLoader(test_dataset_seen, batch_size=args.batch_size, shuffle=False,
                                       num_workers=0, pin_memory=True)
    test_loader_novel = data.DataLoader(test_dataset_novel, batch_size=args.batch_size, shuffle=False,
                                        num_workers=0, pin_memory=True)

    print(len(labeled_train_loader))
    print(len(unlabeled_train_loader))

    # 创建普通模型
    model = build_model(args)

    # 创建EMA模型
    ema_model = build_model(args, ema=True)

    # 最优传输Sinkorn-Knopp算法
    sinkhorn = SinkhornKnopp(args.num_iters_sk, args.epsilon_sk, args.imbalance_factor)

    # 加快卷积层计算速度
    cudnn.benchmark = True

    # 输出参数总数，单位：百万
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 设置两个优化器
    con_optimizer = torch.optim.SGD(
        list(model.backbone.parameters()) +
        list(model.projection_head_target_con.parameters()) +
        list(model.projection_head_coarse_con.parameters()) +
        list(model.projection_head_fine_con.parameters()),
        lr=1,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    cls_optimizer = torch.optim.SGD(
        list(model.target_prototypes.parameters()) +
        list(model.coarse_prototypes.parameters()) +
        list(model.fine_prototypes.parameters()) +
        list(model.projection_head_coarse_cls.parameters()) +
        list(model.projection_head_fine_cls.parameters()),
        lr=0.3,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # 总训练步骤和warmup步骤的数量
    num_training_steps = args.train_iteration * args.epochs
    num_warmup_steps = args.train_iteration * args.warmup_epochs
    # 配置 warmup 和余弦学习率调度器
    con_scheduler = get_cosine_schedule_with_warmup(con_optimizer, num_warmup_steps, num_training_steps,
                                                    warmup_start_lr=args.warmup_start_lr,
                                                    base_lr=args.base_lr,
                                                    final_lr=args.final_lr)
    cls_scheduler = get_cosine_schedule_with_warmup(cls_optimizer, num_warmup_steps, num_training_steps,
                                                    warmup_start_lr=args.warmup_start_lr,
                                                    base_lr=args.base_lr,
                                                    final_lr=args.final_lr)
    # EMA模型优化器
    ema_optimizer = WeightEMA(args, model, ema_model)
    con_criterion = SupConLoss()
    if args.use_cuda:
        print(f'GPU Counts: {torch.cuda.device_count()}')
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')

    writer = SummaryWriter(args.out)

    # 训练循环
    for epoch in range(args.epochs):
        print(f"Epoch : {epoch + 1}")
        losses, con_losses, cls_losses = train(args, labeled_train_loader, unlabeled_train_loader, model, con_criterion,
                                               con_optimizer, con_scheduler, cls_optimizer, cls_scheduler, ema_optimizer,
                                               sinkhorn, args.use_cuda)
        pred_acc_seen = test_seen(args, test_loader_seen, ema_model, epoch)
        pred_acc_novel = test_cluster(args, test_loader_novel, ema_model, epoch, offset=args.no_seen, test_type='novel')
        pred_acc_all = test_cluster(args, test_loader_all, ema_model, epoch, test_type='all')

        # 根据不确定性跟新温度
        if args.uncertain_freq > 0:
            # 重新读取数据集计算不确定性
            if (epoch + 1) % args.uncertain_freq == 0:
                temp_uncertain = uncertain_generator(args, uncertain_loader, ema_model)
                train_labeled_dataset, train_unlabeled_dataset = my_data.get_dataset(temp_uncertain=temp_uncertain)
            # 重新生成DataLoader
            labeled_train_loader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=True)
            unlabeled_train_loader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=True)

        print(f'pred_acc_seen: {pred_acc_seen}')
        print(f"pred_acc_novel: {pred_acc_novel['acc']}")
        print(f"pred_acc_all: {pred_acc_all['acc']}")

        test_acc = pred_acc_all["acc"]

        writer.add_scalar('train/1.total_loss', losses, epoch)
        writer.add_scalar('train/2.con_losses', con_losses, epoch)
        writer.add_scalar('train/3.cls_losses', cls_losses, epoch)
        writer.add_scalar('test/1.acc_seen', pred_acc_seen, epoch)
        writer.add_scalar('test/2.acc_novel', pred_acc_novel['acc'], epoch)
        writer.add_scalar('test/3.acc_all', pred_acc_all['acc'], epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        # save model
        model_to_save = model.module if hasattr(model, "module") else model
        ema_model_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_model_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'con_optimizer': con_optimizer.state_dict(),
            'cls_optimizer': cls_optimizer.state_dict(),
        }, is_best, args.out)

    writer.close()

    print('Best acc:')
    print(best_acc)


def train(args, labeled_train_loader, unlabeled_train_loader, model, con_criterion,
          con_optimizer, con_scheduler, cls_optimizer, cls_scheduler, ema_optimizer, sinkhorn, use_cuda):
    # 记录各项指标
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    con_losses = AverageMeter()
    cls_losses = AverageMeter()
    end = time.time()

    labeled_train_iter = itertools.cycle(labeled_train_loader)
    unlabeled_train_iter = itertools.cycle(unlabeled_train_loader)

    model.train()
    p_bar = tqdm(range(args.train_iteration), desc="Training progress: ")
    for iteration in p_bar:
        # 迭代读取数据，如果迭代器读取完则重置
        # 读取有标签数据
        try:
            (inputs_x, inputs_x2, inputs_xs, inputs_xs2), targets_x, _, temp_x = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_train_loader)
            (inputs_x, inputs_x2, inputs_xs, inputs_xs2), targets_x, _, temp_x = next(labeled_train_iter)
        # 读取无标签数据
        try:
            (inputs_u, inputs_u2, inputs_us, inputs_us2), _, _, temp_u = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_train_loader)
            (inputs_u, inputs_u2, inputs_us, inputs_us2), _, _, temp_u = next(unlabeled_train_iter)
        # 记录读取数据的时间
        data_time.update(time.time() - end)
        # 记录批量大小
        batch_size = inputs_x.size(0)

        # 移动到GPU上
        if use_cuda:
            inputs_x, inputs_x2, inputs_xs, inputs_xs2, targets_x = (inputs_x.cuda(non_blocking=True),
                                                                     inputs_x2.cuda(non_blocking=True),
                                                                     inputs_xs.cuda(non_blocking=True),
                                                                     inputs_xs2.cuda(non_blocking=True),
                                                                     targets_x.cuda(non_blocking=True))
            inputs_u, inputs_u2, inputs_us, inputs_us2 = (inputs_u.cuda(non_blocking=True),
                                                          inputs_u2.cuda(non_blocking=True),
                                                          inputs_us.cuda(non_blocking=True),
                                                          inputs_us2.cuda(non_blocking=True))
            temp_x, temp_u = temp_x.cuda(non_blocking=True), temp_u.cuda(non_blocking=True)

        # 标准化原型的权重
        with torch.no_grad():
            w1 = model.target_prototypes.weight.data.clone()
            w1 = F.normalize(w1, dim=1, p=2)
            model.target_prototypes.weight.copy_(w1)

            w2 = model.coarse_prototypes.weight.data.clone()
            w2 = F.normalize(w2, dim=1, p=2)
            model.coarse_prototypes.weight.copy_(w2)

            w3 = model.fine_prototypes.weight.data.clone()
            w3 = F.normalize(w3, dim=1, p=2)
            model.fine_prototypes.weight.copy_(w3)

        # 获取不同粒度原型的相似度
        coarse_to_target_sim = model.coarse_to_target_sim()
        fine_to_target_sim = model.fine_to_target_sim()

        # 生成分类伪标签和对比学习伪标签
        with torch.no_grad():
            # 无标签数据在三个层级分别输出outputs
            target_outputs_u, coarse_outputs_u, fine_outputs_u = model(inputs_u)
            target_outputs_u2, coarse_outputs_u2, fine_outputs_u2 = model(inputs_u2)

            # 对于有标签数据，获取其在coarse和fine粒度的标签，因为有target粒度的标签
            _, coarse_outputs_x, fine_outputs_x = model(inputs_x)
            _, coarse_outputs_x2, fine_outputs_x2 = model(inputs_x2)

            # 两个增强视角之间交换使用标签
            end = time.time()

            # 交换无标签数据两个视角target级别的预测
            targets_u = sinkhorn(target_outputs_u2)
            targets_u2 = sinkhorn(target_outputs_u)

            # 交换无标签数和有标签数据两个视角coarse级别的预测
            coarse_targets_u = sinkhorn(coarse_outputs_u2)
            coarse_targets_u2 = sinkhorn(coarse_outputs_u)
            coarse_targets_x = sinkhorn(coarse_outputs_x2)
            coarse_targets_x2 = sinkhorn(coarse_outputs_x)

            # 交换无标签数和有标签数据据两个视角fine级别的预测
            fine_targets_u = sinkhorn(fine_outputs_u2)
            fine_targets_u2 = sinkhorn(fine_outputs_u)
            fine_targets_x = sinkhorn(fine_outputs_x2)
            fine_targets_x2 = sinkhorn(fine_outputs_x)

            # 目标梯度两个视角的硬标签
            _, hard_targets_u = torch.max(targets_u, dim=-1)
            _, hard_targets_u2 = torch.max(targets_u2, dim=-1)
            # 计算coarse级别的硬伪标签
            _, coarse_hard_targets_u = torch.max(coarse_targets_u, dim=-1)
            _, coarse_hard_targets_u2 = torch.max(coarse_targets_u2, dim=-1)
            _, coarse_hard_targets_x = torch.max(coarse_targets_x, dim=-1)
            _, coarse_hard_targets_x2 = torch.max(coarse_targets_x2, dim=-1)
            # 计算fine级别的硬伪标签
            _, fine_hard_targets_u = torch.max(fine_targets_u, dim=-1)
            _, fine_hard_targets_u2 = torch.max(fine_targets_u2, dim=-1)
            _, fine_hard_targets_x = torch.max(fine_targets_x, dim=-1)
            _, fine_hard_targets_x2 = torch.max(fine_targets_x2, dim=-1)

            '''
                两种标签再加权SupCon损失
            '''
            # 粗粒度两个视角的硬标签
            con_coarse_labels_view1 = torch.cat([coarse_hard_targets_x, coarse_hard_targets_u])
            con_coarse_labels_view2 = torch.cat([coarse_hard_targets_x2, coarse_hard_targets_u2])
            # 细粒度两个视角的硬标签
            con_fine_labels_view1 = torch.cat([fine_hard_targets_x, fine_hard_targets_u])
            con_fine_labels_view2 = torch.cat([fine_hard_targets_x2, fine_hard_targets_u2])
            # 目标粒度两个视角的硬标签
            con_target_labels_view1 = torch.cat([targets_x, hard_targets_u])
            con_target_labels_view2 = torch.cat([targets_x, hard_targets_u2])

            # 计算其他粒度logits映射到目标粒度logits
            coarse_to_target_labels_view1 = torch.mm(torch.cat([coarse_targets_x, coarse_targets_u]),
                                                     coarse_to_target_sim)
            coarse_to_target_labels_view2 = torch.mm(torch.cat([coarse_targets_x2, coarse_targets_u2]),
                                                     coarse_to_target_sim)
            fine_to_target_labels_view1 = torch.mm(torch.cat([fine_targets_x, fine_targets_u]), fine_to_target_sim)
            fine_to_target_labels_view2 = torch.mm(torch.cat([fine_targets_x2, fine_targets_u2]), fine_to_target_sim)

            # 生成映射后的硬伪标签
            _, coarse_to_target_labels_view1 = torch.max(coarse_to_target_labels_view1, dim=-1)
            _, coarse_to_target_labels_view2 = torch.max(coarse_to_target_labels_view2, dim=-1)
            _, fine_to_target_labels_view1 = torch.max(fine_to_target_labels_view1, dim=-1)
            _, fine_to_target_labels_view2 = torch.max(fine_to_target_labels_view2, dim=-1)

        # Compute classification loss

        # 将有标签数据的标签转换为one-hot标签
        targets_x = torch.zeros(batch_size, args.no_class, device=targets_x.device) \
            .scatter_(1, targets_x.view(-1, 1).long(), 1)

        targets_u_novel = targets_u[:, args.no_seen:]
        max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
        hard_novel_idx1 = torch.where(max_pred_novel >= args.threshold)[0]

        targets_u2_novel = targets_u2[:, args.no_seen:]
        max_pred2_novel, _ = torch.max(targets_u2_novel, dim=-1)
        hard_novel_idx2 = torch.where(max_pred2_novel >= args.threshold)[0]

        # 对新类根据索引转换为硬伪标签
        targets_u[hard_novel_idx1] = targets_u[hard_novel_idx1].ge(args.threshold).float()
        targets_u2[hard_novel_idx2] = targets_u2[hard_novel_idx2].ge(args.threshold).float()

        # mixup
        # 拼接所有输入
        cls_all_inputs = torch.cat([inputs_x, inputs_u, inputs_x2, inputs_u2], dim=0)

        # 拼接所有目标粒度的标签 (4*batch_size, target_num_class) 既有软伪标签，又有硬伪标签
        all_targets = torch.cat([targets_x, targets_u, targets_x, targets_u2], dim=0)

        # 拼接所有粗粒度的标签 (4*bz, concept_num_class) # hard label
        all_coarse_targets = torch.cat([coarse_hard_targets_x, coarse_hard_targets_u,
                                        coarse_hard_targets_x2, coarse_hard_targets_u2], dim=0)
        # 转换为one-hot编码
        all_coarse_targets = torch.zeros(all_coarse_targets.shape[0], args.num_coarse).cuda(non_blocking=True). \
            scatter_(1, all_coarse_targets.view(-1, 1).long(), 1)

        # 拼接所有细粒度的标签
        all_fine_targets = torch.cat([fine_hard_targets_x, fine_hard_targets_u,
                                      fine_hard_targets_x2, fine_hard_targets_u2], dim=0)
        # 转换为one-hot编码
        all_fine_targets = torch.zeros(all_fine_targets.shape[0], args.num_fine).cuda(non_blocking=True). \
            scatter_(1, all_fine_targets.view(-1, 1).long(), 1)

        # 拼接所有温度参数
        all_temp = torch.cat([temp_x, temp_u, temp_x, temp_u], dim=0)

        # mixup比例
        mixup_lambda = np.random.beta(args.alpha, args.alpha)
        # 获取随机排列
        idx = torch.randperm(cls_all_inputs.size(0))
        # 获取原始输入和打乱后的输入
        input_a, input_b = cls_all_inputs, cls_all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        coarse_target_a, coarse_target_b = all_coarse_targets, all_coarse_targets[idx]
        fine_target_a, fine_target_b = all_fine_targets, all_fine_targets[idx]
        temp_a, temp_b = all_temp, all_temp[idx]
        # 根据lambda进行mixup混合
        mixed_input = mixup_lambda * input_a + (1 - mixup_lambda) * input_b
        mixed_target = mixup_lambda * target_a + (1 - mixup_lambda) * target_b
        mixed_temp = mixup_lambda * temp_a + (1 - mixup_lambda) * temp_b
        mixed_coarse_target = mixup_lambda * coarse_target_a + (1 - mixup_lambda) * coarse_target_b
        mixed_fine_target = mixup_lambda * fine_target_a + (1 - mixup_lambda) * fine_target_b

        # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        temp_target_logits, temp_coarse_logits, temp_fine_logits = model(mixed_input[0])
        target_logits = [temp_target_logits]
        coarse_logits = [temp_coarse_logits]
        fine_logits = [temp_fine_logits]

        for inputs in mixed_input[1:]:
            temp_target_logits, temp_coarse_logits, temp_fine_logits = model(inputs)
            target_logits.append(temp_target_logits)
            coarse_logits.append(temp_coarse_logits)
            fine_logits.append(temp_fine_logits)

        # put interleaved samples back
        target_logits = interleave(target_logits, batch_size)
        target_logits = torch.cat(target_logits, dim=0)

        coarse_logits = interleave(coarse_logits, batch_size)
        coarse_logits = torch.cat(coarse_logits, dim=0)

        fine_logits = interleave(fine_logits, batch_size)
        fine_logits = torch.cat(fine_logits, dim=0)

        # 计算分类交叉熵损失

        # cross_entropy loss 目标粒度的logits和labels
        preds = F.softmax(target_logits / mixed_temp.unsqueeze(1), dim=1)
        preds = torch.clamp(preds, min=1e-8)
        preds = torch.log(preds)
        target_loss = - torch.mean(torch.sum(mixed_target * preds, dim=1))
        # cross_entropy loss for coarse level
        coarse_preds = F.softmax(coarse_logits / mixed_temp.unsqueeze(1), dim=1)
        coarse_preds = torch.clamp(coarse_preds, min=1e-8)
        coarse_preds = torch.log(coarse_preds)
        coarse_loss = - torch.mean(torch.sum(mixed_coarse_target * coarse_preds, dim=1))
        # cross_entropy loss for fine level
        fine_preds = F.softmax(fine_logits / mixed_temp.unsqueeze(1), dim=1)
        fine_preds = torch.clamp(fine_preds, min=1e-8)
        fine_preds = torch.log(fine_preds)
        fine_loss = - torch.mean(torch.sum(mixed_fine_target * fine_preds, dim=1))

        # cross_entropy loss for coarse to target
        coarse_to_target_logits = coarse_logits @ coarse_to_target_sim
        coarse_to_target_preds = F.softmax(coarse_to_target_logits / mixed_temp.unsqueeze(1), dim=1)
        coarse_to_target_preds = torch.clamp(coarse_to_target_preds, min=1e-8)
        coarse_to_target_preds = torch.log(coarse_to_target_preds)
        # 使用target的label和从coarse粒度对齐后的logits
        coarse2target_loss = - torch.mean(torch.sum(mixed_target * coarse_to_target_preds, dim=1))
        # cross_entropy loss for fine to target
        fine_to_target_logits = fine_logits @ fine_to_target_sim
        fine_to_target_preds = F.softmax(fine_to_target_logits / mixed_temp.unsqueeze(1), dim=1)
        fine_to_target_preds = torch.clamp(fine_to_target_preds, min=1e-8)
        fine_to_target_preds = torch.log(fine_to_target_preds)
        # 使用target的label和从fine粒度对齐后的logits
        fine2target_loss = - torch.mean(torch.sum(mixed_target * fine_to_target_preds, dim=1))

        # strong view
        strong_inputs = torch.cat([inputs_xs, inputs_us, inputs_xs2, inputs_us2], dim=0)
        strong_inputs = interleave(torch.split(strong_inputs, batch_size), batch_size)

        temp_target_logits_us, temp_coarse_logits_us, temp_fine_logits_us, \
            temp_target_features, temp_coarse_features, temp_fine_features = model(strong_inputs[0], contrastive_learning=True)
        target_logits_us = [temp_target_logits_us]
        coarse_logits_us = [temp_coarse_logits_us]
        fine_logits_us = [temp_fine_logits_us]
        target_features = [temp_target_features]
        coarse_features = [temp_coarse_features]
        fine_features = [temp_fine_features]
        for inputs in strong_inputs[1:]:
            temp_target_logits_us, temp_coarse_logits_us, temp_fine_logits_us, \
                temp_target_features, temp_coarse_features, temp_fine_features = model(inputs, contrastive_learning=True)
            target_logits_us.append(temp_target_logits_us)
            coarse_logits_us.append(temp_coarse_logits_us)
            fine_logits_us.append(temp_fine_logits_us)
            target_features.append(temp_target_features)
            coarse_features.append(temp_coarse_features)
            fine_features.append(temp_fine_features)
        target_logits_us = torch.cat(interleave(target_logits_us, batch_size), dim=0)
        coarse_logits_us = torch.cat(interleave(coarse_logits_us, batch_size), dim=0)
        fine_logits_us = torch.cat(interleave(fine_logits_us, batch_size), dim=0)
        target_features = torch.cat(interleave(target_features, batch_size), dim=0)
        coarse_features = torch.cat(interleave(coarse_features, batch_size), dim=0)
        fine_features = torch.cat(interleave(fine_features, batch_size), dim=0)

        target_logits_us1 = target_logits_us[batch_size:2 * batch_size]
        target_preds_us1 = F.softmax(target_logits_us1 / temp_u.unsqueeze(1), dim=1)
        target_preds_us1 = torch.clamp(target_preds_us1, min=1e-8)
        target_preds_us1 = torch.log(target_preds_us1)
        target_logits_us2 = target_logits_us[3 * batch_size:4 * batch_size]
        target_preds_us2 = F.softmax(target_logits_us2 / temp_u.unsqueeze(1), dim=1)
        target_preds_us2 = torch.clamp(target_preds_us2, min=1e-8)
        target_preds_us2 = torch.log(target_preds_us2)
        # 用u1和u2的目标粒度预测做伪标签，与强增强的logits计算交叉熵, 两个视角都使用，再做一个加权平均
        target_us_loss = -0.5 * (torch.mean(torch.sum(targets_u * target_preds_us1, dim=1)) +
                                 torch.mean(torch.sum(targets_u2 * target_preds_us2, dim=1)))

        coarse_logits_us1 = coarse_logits_us[batch_size:2 * batch_size]
        coarse_preds_us1 = F.softmax(coarse_logits_us1 / temp_u.unsqueeze(1), dim=1)
        coarse_preds_us1 = torch.clamp(coarse_preds_us1, min=1e-8)
        coarse_preds_us1 = torch.log(coarse_preds_us1)
        coarse_logits_us2 = coarse_logits_us[3 * batch_size:4 * batch_size]
        coarse_preds_us2 = F.softmax(coarse_logits_us2 / temp_u.unsqueeze(1), dim=1)
        coarse_preds_us2 = torch.clamp(coarse_preds_us2, min=1e-8)
        coarse_preds_us2 = torch.log(coarse_preds_us2)
        # 用u1和u2的粗粒度预测做伪标签，与强增强的logits计算交叉熵, 两个视角都使用，再做一个加权平均
        coarse_us_loss = -0.5 * (torch.mean(torch.sum(coarse_targets_u * coarse_preds_us1, dim=1)) +
                                 torch.mean(torch.sum(coarse_targets_u2 * coarse_preds_us2, dim=1)))

        fine_logits_us1 = fine_logits_us[batch_size:2 * batch_size]
        fine_preds_us1 = F.softmax(fine_logits_us1 / temp_u.unsqueeze(1), dim=1)
        fine_preds_us1 = torch.clamp(fine_preds_us1, min=1e-8)
        fine_preds_us1 = torch.log(fine_preds_us1)
        fine_logits_us2 = fine_logits_us[3 * batch_size:4 * batch_size]
        fine_preds_us2 = F.softmax(fine_logits_us2 / temp_u.unsqueeze(1), dim=1)
        fine_preds_us2 = torch.clamp(fine_preds_us2, min=1e-8)
        fine_preds_us2 = torch.log(fine_preds_us2)
        # 用u1和u2的细粒度预测做伪标签，与强增强的logits计算交叉熵, 两个视角都使用，再做一个加权平均
        fine_us_loss = -0.5 * (torch.mean(torch.sum(fine_targets_u * fine_preds_us1, dim=1)) +
                               torch.mean(torch.sum(fine_targets_u2 * fine_preds_us2, dim=1)))

        cls_loss = ((coarse2target_loss + fine2target_loss) * 0.5
                    + (target_us_loss + coarse_us_loss + fine_us_loss) * 0.33
                    + target_loss + coarse_loss + fine_loss) * 0.5

        # Compute SupCon Loss
        # 处理为计算SupCon的格式
        f1, f2 = torch.split(target_features, [2 * batch_size, 2 * batch_size], dim=0)
        target_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        f1, f2 = torch.split(coarse_features, [2 * batch_size, 2 * batch_size], dim=0)
        coarse_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        f1, f2 = torch.split(fine_features, [2 * batch_size, 2 * batch_size], dim=0)
        fine_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # 根据特征和各粒度的伪标签计算SupCon Loss
        con_target_loss = 0.5 * (con_criterion(target_features, con_target_labels_view1) +
                                 con_criterion(target_features, con_target_labels_view2))

        con_coarse_loss = 0.5 * (con_criterion(coarse_features, con_coarse_labels_view1) +
                                 con_criterion(coarse_features, con_coarse_labels_view2))

        con_fine_loss = 0.5 * (con_criterion(fine_features, con_fine_labels_view1) +
                               con_criterion(fine_features, con_fine_labels_view2))

        con_coarse_to_target_loss = 0.5 * (con_criterion(target_features, coarse_to_target_labels_view1) +
                                           con_criterion(target_features, coarse_to_target_labels_view2))

        con_fine_to_target_loss = 0.5 * (con_criterion(target_features, fine_to_target_labels_view1) +
                                         con_criterion(target_features, fine_to_target_labels_view2))

        con_loss = (con_target_loss + con_coarse_loss + con_fine_loss +
                    (con_coarse_to_target_loss + con_fine_to_target_loss) * 0.5) * 0.25

        total_loss = con_loss + cls_loss

        con_optimizer.zero_grad()
        cls_optimizer.zero_grad()
        total_loss.backward()
        con_optimizer.step()
        cls_optimizer.step()

        ema_optimizer.step()
        con_scheduler.step()
        cls_scheduler.step()

        losses.update(total_loss.item())
        con_losses.update(con_loss.item())
        cls_losses.update(cls_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # 更新进度条
        p_bar.set_postfix({"total_loss": losses.avg, "con_losses": con_losses.avg, "cls_losses": cls_losses.avg,
                           "data_time": data_time.avg, "batch_time": batch_time.avg, "iter": iteration + 1})
    p_bar.close()
    return losses.avg, con_losses.avg, cls_losses.avg


if __name__ == '__main__':
    main()
