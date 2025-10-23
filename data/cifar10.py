import numpy as np
from torchvision import datasets
from PIL import Image


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexes, temperature=None, temp_uncertain=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        if temperature is not None:
            self.temp = temperature * np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncertain is not None:
            self.temp[temp_uncertain['index']] = temp_uncertain['uncertain']

        if indexes is not None:
            indexes = np.array(indexes)
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]
            self.temp = self.temp[indexes]
            self.indexes = indexes
        else:
            self.indexes = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexes[index], self.temp[index]


class CIFAR10TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=True, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexes = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexes.extend(idx)
            indexes = np.array(indexes)
            self.data = self.data[indexes]
            self.targets = self.targets[indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10UNCERTAIN(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.transform(img)
            img4 = self.transform(img)
            img5 = self.transform(img)
            img6 = self.transform(img)
            img7 = self.transform(img)
            img8 = self.transform(img)
            img9 = self.transform(img)
            img10 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, target, self.indexs[index]
