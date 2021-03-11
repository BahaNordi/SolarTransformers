import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)
# train_dir = '/home/baha/codes/solar_data/InfraredSolarModules/train'
# test_dir = '/home/baha/codes/solar_data/InfraredSolarModules/val'

train_dir = '/content/gdrive/MyDrive/Colab_Notebooks/jua_solar/solar_data/InfraredSolarModules/train'
test_dir = '/content/gdrive/MyDrive/Colab_Notebooks/jua_solar/solar_data/InfraredSolarModules/val'


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([

        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.7, 1.0)),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                                           saturation=0.1, hue=0.05)]), p=0.3),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur([3], sigma=(0.1, 2.0))]), p=0.3),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    else:
        trainset = ImageFolder(train_dir, transform=transform_train)
        testset = ImageFolder(test_dir, transform=transform_test) if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    weights = make_weights_for_balanced_classes(trainset.imgs, len(trainset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    n = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = n / float(count[i])
    weight_per_class[7] = 10.
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight
