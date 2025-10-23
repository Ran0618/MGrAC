import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision import datasets, transforms
import torch
import pickle
import os
import math
from .rand_augment import MyTransform, SolarizeCallable, EqualizeCallable, TwoCropTransform
from utils.split import x_u_split_seen_novel
from .cifar10 import CIFAR10SSL, CIFAR10TEST, CIFAR10UNCERTAIN
from .cifar100 import CIFAR100SSL, CIFAR100TEST, CIFAR100UNCERTAIN

# normalization parameters
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
tinyimagenet_mean, tinyimagenet_std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataset_class(args):
    if args.dataset == 'cifar10':
        return CIFAR10DATASET(args)
    elif args.dataset == 'cifar100':
        return CIFAR100DATASET(args)


class CIFAR10DATASET:
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(32, (0.5, 1.0)),
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            SolarizeCallable(p=0.1),
            EqualizeCallable(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.labeled_percent,
                                                                        args.no_class, list(range(0, args.no_seen)),
                                                                        list(range(args.no_seen, args.no_class)),
                                                                        args.imbalance_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncertain=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR10SSL(self.data_root, train_labeled_idxs, train=True,
                                           transform=MyTransform(cifar10_mean, cifar10_std, self.transform_train),
                                           temperature=self.temperature)
        train_unlabeled_dataset = CIFAR10SSL(self.data_root, train_unlabeled_idxs, train=True,
                                             transform=MyTransform(cifar10_mean, cifar10_std, self.transform_train),
                                             temperature=self.temperature, temp_uncertain=temp_uncertain)

        if temp_uncertain is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        uncertain_dataset = CIFAR10UNCERTAIN(self.data_root, train_unlabeled_idxs, train=True,
                                             transform=self.transform_train)
        test_dataset_all = CIFAR10TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        test_dataset_seen = CIFAR10TEST(self.data_root, train=False, transform=self.transform_val, download=False,
                                        labeled_set=list(range(0, self.no_seen)))
        test_dataset_novel = CIFAR10TEST(self.data_root, train=False, transform=self.transform_val, download=False,
                                         labeled_set=list(range(self.no_seen, self.no_class)))

        return train_labeled_dataset, train_unlabeled_dataset, uncertain_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel


class CIFAR100DATASET:
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(32, (0.5, 1.0)),
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            SolarizeCallable(p=0.1),
            EqualizeCallable(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.labeled_percent,
                                                                        args.no_class, list(range(0, args.no_seen)),
                                                                        list(range(args.no_seen, args.no_class)),
                                                                        args.imbalance_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncertain=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR100SSL(self.data_root, train_labeled_idxs, train=True,
                                            transform=MyTransform(cifar100_mean, cifar100_std, self.transform_train),
                                            temperature=self.temperature)
        train_unlabeled_dataset = CIFAR100SSL(self.data_root, train_unlabeled_idxs, train=True,
                                              transform=MyTransform(cifar100_mean, cifar100_std, self.transform_train),
                                              temperature=self.temperature, temp_uncertain=temp_uncertain)

        if temp_uncertain is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        uncertain_dataset = CIFAR100UNCERTAIN(self.data_root, train_unlabeled_idxs, train=True,
                                              transform=self.transform_train)
        test_dataset_all = CIFAR100TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        test_dataset_seen = CIFAR100TEST(self.data_root, train=False, transform=self.transform_val, download=False,
                                         labeled_set=list(range(0, self.no_seen)))
        test_dataset_novel = CIFAR100TEST(self.data_root, train=False, transform=self.transform_val, download=False,
                                          labeled_set=list(range(self.no_seen, self.no_class)))

        return train_labeled_dataset, train_unlabeled_dataset, uncertain_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel
