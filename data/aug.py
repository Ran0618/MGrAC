from torchvision.transforms import transforms
from data.gaussian_blur import SimCLRGaussianBlur
from data.gaussian_blur import MoCoGaussianBlur
from torchvision import transforms


class ContrastiveLearningTransform:
    @staticmethod
    def get_transform(args, size=224, s=1):

        # Set normalize for each dataset
        if args.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif args.dataset == 'imagenet':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise ValueError('dataset not supported: {}'.format(args.dataset))
        normalize = transforms.Normalize(mean=mean, std=std)

        """Return a set of data augmentation transformations"""

        # MoCo style data augmentation
        moco_augmentation = [
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([MoCoGaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        # SimCLR style data augmentation
        simclr_augmentation = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            SimCLRGaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            normalize
        ]

        data_transforms = transforms.Compose(simclr_augmentation)

        return data_transforms
