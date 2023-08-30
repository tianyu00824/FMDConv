from __future__ import print_function

import matplotlib.pyplot as plt

import argparse
import os
import shutil
import time
import random
from math import cos, pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models as customized_models
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import Logger, AverageMeter, accuracy, mkdir_p, get_temperature, init_distributed_mode, get_dist_info
from torch.cuda.amp.grad_scaler import GradScaler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='batch size during training (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='batch size during testing (default: 100)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
parser.add_argument('--lr-decay', type=str, default=None,
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout (default: 0')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

# Miscs
parser.add_argument('--manualSeed', default=None, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#Device options
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--gpu_ids", default=-1, type=int)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--use_amp', default=False, action='store_true', help='use automatic mixed precision')

#ODConv options
parser.add_argument('--temp_epoch', type=int, default=10, help='number of epochs for temperature annealing')
parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')
parser.add_argument('--reduction', type=float, default=0.0625, help='reduction ratio used in the attention module')
parser.add_argument('--kernel_num', type=int, default=1, help='number of convolutional kernels in ODConv')

parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup or not')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    cudnn.benchmark = True

    init_distributed_mode(args)
    _, args.world_size = get_dist_info()
    args.gpu_ids = range(args.world_size)
    print(1)
    print(args.gpu_ids)
    args.train_batch = args.train_batch // args.world_size
    args.local_rank = torch.cuda.current_device()
    print("World Size", args.world_size)
    print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](dropout=args.dropout,
                                       reduction=args.reduction,
                                       kernel_num=args.kernel_num)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.distributed:
        model = DDP(model.cuda(), device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda(args.gpu_ids[0]), device_ids=args.gpu_ids)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.dataset == 'cifar10':
        numclasses = 10  # 十分类
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True,  # 下载训练集数据
                                                transform=transforms.Compose([  # 数据预处理
                                                    transforms.Pad(4),  # 图像填充上下左右4个pixel
                                                    transforms.RandomCrop(32),  # 图像随机裁剪 32*32 大小图片
                                                    transforms.RandomHorizontalFlip(),  # 图像随机水平翻转
                                                    transforms.ToTensor(),  # 将图片转换成tensor
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))
                                                ]))  # 正则化：降低模型复杂度
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0)
        # 加载训练集数据：batch_size一次训练板块大小，shuffle打乱，num)worker:创建多个num_worker加载batch数据
        testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True,  # 下载测试集数据
                                               transform=transforms.Compose([  # 数据预处理
                                                   transforms.ToTensor(),  # 将图片转换成tensor
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010))
                                               ]))  # 正则化：降低模型复杂度
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)
    elif args.dataset == 'cifar100':
        numclasses = 100  # 一百分类
        trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True,  # 下载训练集数据
                                                 transform=transforms.Compose([  # 数据预处理
                                                     transforms.Pad(4),  # 图像填充上下左右4个pixel
                                                     transforms.RandomCrop(32),  # 图像随机裁剪 32*32 大小图片
                                                     transforms.RandomHorizontalFlip(),  # 图像随机水平翻转
                                                     transforms.ToTensor(),  # 将图片转换成tensor
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010))
                                                 ]))  # 正则化：降低模型复杂度
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0)
        # 加载训练集数据：batch_size一次训练板块大小，shuffle打乱，num)worker:创建多个num_worker加载batch数据
        testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True,  # 下载测试集数据
                                                transform=transforms.Compose([  # 数据预处理
                                                    transforms.ToTensor(),  # 将图片转换成tensor
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))
                                                ]))  # 正则化：降低模型复杂度
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    #trainset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(224),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))
    #train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    #train_loader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=args.train_batch,
    #                                           sampler=train_sampler,
    #                                           num_workers=args.workers,
    #                                           pin_memory=True,
    #                                           drop_last=False)

    #val_dataset = datasets.ImageFolder(
    #    valdir,
    #    transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))

    #val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    #val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                         batch_size=args.test_batch,
    #                                         num_workers=args.workers,
    #                                         pin_memory=True,
    #                                         sampler=val_sampler,
    #                                         drop_last=False)

    train_loader_len, val_loader_len = len(train_loader), len(val_loader)



    #classes = ('plane', 'car', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #with open('/hy-tmp/pycharm_1/cifar10/cifar-10-batches-py/data_batch_1', 'rb') as f:
    #     file = f.read()
    #image1 = [int(str(item).encode('ascii'),16)for item in file[16 : 16+784]]
    #print(image1)

    #import numpy as np
    #image_np = np.array(image1,dtype=np.uint8).reshape(28,28,1)
    #print(image_np.shape)

    #for i, (images, _) in enumerate(train_loader):
        #a = torch.mean(images, dim=2)
    #    torch.save(images[0], "aaa.pth")
        #print(i)
        #print(images.numpy().shape)
        #imshow(images[0])
        #plt.imshow((out * 255).astype(np.uint8))

        #pic = toPIL(img)
        #pic.save('random2.jpg')

    #    break

    #for i, (images, _) in enumerate(train_loader):
    #    torch.save(torchvision.utils.make_grid(images), "bbb.pth")
        #imshow(torchvision.utils.make_grid(images))
    #    break





    if args.evaluate:
        print('\nEvaluation only')
        with torch.no_grad():
            test_loss, test_acc = test(val_loader, val_loader_len, model, criterion, use_cuda)
        print('Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    if args.resume:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if hasattr(model.module, "net_update_temperature"):
        temp = get_temperature(0, start_epoch, train_loader_len,
                               temp_epoch=args.temp_epoch, temp_init=args.temp_init)
        model.module.net_update_temperature(temp)

    if args.use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        #train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, 0, train_loader_len)
        lr = optimizer.param_groups[0]['lr']

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion,
                                      optimizer, epoch, use_cuda, scaler)

        with torch.no_grad():
            test_loss, test_acc = test(val_loader, val_loader_len, model, criterion, use_cuda)

        if args.local_rank == 0:
            # append logger file
            logger.append([lr, train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc.cpu().data > best_acc
            best_acc = max(test_acc.cpu().data, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint)

    logger.close()


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, use_cuda, scaler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    start_time = time.time()
    temp = 1.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # update temperature of ODConv
        if epoch < args.temp_epoch and hasattr(model.module, 'net_update_temperature'):
            temp = get_temperature(batch_idx + 1, epoch, train_loader_len,
                                   temp_epoch=args.temp_epoch, temp_init=args.temp_init)
            model.module.net_update_temperature(temp)

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # normal forward
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        if args.distributed:
            torch.distributed.barrier()
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % args.print_freq == 0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | '
                  'top5: {top5: .4f} | Temp: {temp: .4f} | Total Time: {time: .2f}'.format(
                        batch=batch_idx + 1,
                        size=train_loader_len,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        temp=temp,
                        time=time.time()-start_time
            ))
    return losses.avg, top1.avg


def test(val_loader, val_loader_len, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        if args.distributed:
            torch.distributed.barrier()
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        #torch.cuda.synchronize()
        #batch_time.update(time.time() - end)
        #end = time.time()

        # plot progress
        if (batch_idx % args.print_freq == 0) or batch_idx == val_loader_len-1:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | '
                   'top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=val_loader_len,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def adjust_learning_rate2(optimizer, epoch, iteration, iter_per_epoch):
    current_iter = iteration + epoch * iter_per_epoch
    max_iter = args.epochs * iter_per_epoch

    if args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * current_iter / max_iter)) / 2
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))


    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imshow(img):
    img=img/2+0.5
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.show()


if __name__ == '__main__':
    main()