import numpy as np
import math


def x_u_split_seen_novel(labels, lbl_percent, num_classes, lbl_set, unlbl_set, imb_factor):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        img_max = len(idx)
        num = img_max * ((1 / imb_factor) ** (i / (num_classes - 1.0)))
        idx = idx[:int(num)]
        n_lbl_sample = math.ceil(len(idx) * (lbl_percent / 100))
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:])
        elif i in unlbl_set:
            unlabeled_idx.extend(idx)

    return labeled_idx, unlabeled_idx


def my_func():
    print("Hello")
