import sys

import torch
import torch.nn as nn
from . import resnet_cifar, resnet_cifar_v2


def build_model(args, ema=False):
    if args.dataset in ['cifar10', 'cifar100']:
        if args.arch == 'resnet18':
            if args.mode == 'v2':
                model = resnet_cifar_v2.resnet18(no_class=args.no_class, num_coarse=args.num_coarse,
                                                 num_fine=args.num_fine,
                                                 hidden_mlp=args.hidden_mlp, feat_dim=args.feat_dim)
            else:
                model = resnet_cifar.resnet18(no_class=args.no_class, num_coarse=args.num_coarse,
                                              num_fine=args.num_fine,
                                              hidden_mlp=args.hidden_mlp, feat_dim=args.feat_dim)
        elif args.arch == 'resnet50':
            if args.arch == 'resnet18':
                if args.mode == 'v2':
                    model = resnet_cifar_v2.resnet50(no_class=args.no_class, num_coarse=args.num_coarse,
                                                     num_fine=args.num_fine,
                                                     hidden_mlp=args.hidden_mlp, feat_dim=args.feat_dim)
                else:
                    model = resnet_cifar.resnet50(no_class=args.no_class, num_coarse=args.num_coarse,
                                                  num_fine=args.num_fine,
                                                  hidden_mlp=args.hidden_mlp, feat_dim=args.feat_dim)
        else:
            print("No such model")
            sys.exit(0)
    else:
        print("No such dataset")
        sys.exit(0)

    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model
