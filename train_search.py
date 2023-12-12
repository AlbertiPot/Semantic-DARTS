import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from model_search import Network
from architect import Architect
from utils import MaskMSE, SamplesSaver

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data',
                    type=str,
                    default='./data/cifar/cifar10',
                    help='location of the data corpus')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10',
                    help='specify dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.025,
                    help='init learning rate')
parser.add_argument('--learning_rate_min',
                    type=float,
                    default=0.001,
                    help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay',
                    type=float,
                    default=3e-4,
                    help='weight decay')
parser.add_argument('--report_freq',
                    type=float,
                    default=50,
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels',
                    type=int,
                    default=16,
                    help='num of init channels')
parser.add_argument('--layers',
                    type=int,
                    default=8,
                    help='total number of layers')
parser.add_argument('--model_path',
                    type=str,
                    default='saved_models',
                    help='path to save the model')
parser.add_argument('--cutout',
                    action='store_true',
                    default=False,
                    help='use cutout')
parser.add_argument('--cutout_length',
                    type=int,
                    default=16,
                    help='cutout length')
parser.add_argument('--drop_path_prob',
                    type=float,
                    default=0.3,
                    help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
parser.add_argument('--train_portion',
                    type=float,
                    default=0.5,
                    help='portion of training data')
parser.add_argument('--unrolled',
                    action='store_true',
                    default=False,
                    help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate',
                    type=float,
                    default=3e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',
                    type=float,
                    default=1e-3,
                    help='weight decay for arch encoding')
parser.add_argument('--task',
                    type=str,
                    default='cls',
                    help='specify pretext task')
parser.add_argument('--mask_ratio',
                    type=float,
                    default=0.8,
                    help='mask ratio of original image')
parser.add_argument('--patch_size',
                    type=int,
                    default=4,
                    help='patch size for masking image')
parser.add_argument('--visual',
                    action='store_true',
                    default=False,
                    help='visualize intermediate feature')

args = parser.parse_args()

assert args.dataset in ['cifar10', 'cifar100'
                        ], "Dataset '{}' not supported".format(args.dataset)
assert args.task in ['cls', 'rec', 'mask_rec',
                     'cls_mask'], "Task '{}' not supported".format(args.task)

if args.dataset == 'cifar100':
    CIFAR_CLASSES = 100
    DATA_STATS = (-1.9, 2.03)
    data_folder = 'cifar-100-python'
elif args.dataset == 'cifar10':
    CIFAR_CLASSES = 10
    DATA_STATS = (-1.99, 2.13)
    data_folder = 'cifar-10-batches-py'

args.save = '/data/gbc/workspace/rdarts/results/search-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

tb_writer = SummaryWriter(args.save)
samples_saver = None
if args.task == 'cls_mask' or 'rec' in args.task:
    samples_saver = SamplesSaver(args.task, args.save)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.task == 'rec':
        criterion = nn.MSELoss()
    elif args.task == 'cls':
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'mask_rec':
        criterion = MaskMSE()
    elif args.task == 'cls_mask':
        c1 = nn.CrossEntropyLoss()
        c2 = MaskMSE()
        criterion = [c1.cuda(), c2.cuda()]

    if type(criterion) is not list:
        criterion = criterion.cuda()

    model = Network(args.init_channels,
                    CIFAR_CLASSES,
                    args.layers,
                    criterion,
                    DATA_STATS,
                    task=args.task,
                    patch_size=args.patch_size)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #  prepare dataset
    if args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(
            args)
    elif args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)

    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=args.data,
                                   train=True,
                                   download=False,
                                   transform=train_transform)
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(root=args.data,
                                  train=True,
                                  download=False,
                                  transform=train_transform)

    if args.dataset == 'cifar10' and args.visual is True:
        _, test_transform = utils._data_transforms_cifar10(args)
        visual_data = dset.CIFAR10(root=args.data,
                                   train=False,
                                   download=False,
                                   transform=test_transform)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(args.seed)
        visual_idx = torch.randperm(len(visual_data), generator=g_cpu)[:100]
        logging.info('visual image index = {}'.format(visual_idx))
        feature_dict = {}

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=8)

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True,
        num_workers=8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    best_metric = 0
    best_epoch = 0
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                     architect, criterion, optimizer, lr,
                                     epoch)

        if 'cls' in args.task:
            logging.info('train_loss %e, train_acc %f', train_obj, train_acc)
            tb_writer.add_scalar('train/epoch_acc', train_acc, epoch)
        elif 'rec' in args.task:
            logging.info('train_loss %e', train_obj)
        tb_writer.add_scalar('train/epoch_loss', train_obj, epoch)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        logging.info(F.softmax(model.alphas_normal, dim=-1))
        logging.info(F.softmax(model.alphas_reduce, dim=-1))

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)

        if 'cls' in args.task:
            logging.info('valid_loss %e, valid_acc %f', valid_obj, valid_acc)
            tb_writer.add_scalar('valid/epoch_acc', valid_acc, epoch)
            cur_metric = valid_acc
        elif 'rec' in args.task:
            logging.info('valid_loss %e', valid_obj)
            cur_metric = 1 / (valid_obj + 1e-6)

        tb_writer.add_scalar('valid/epoch_loss', valid_obj, epoch)

        mem_usage_bytes = torch.cuda.max_memory_allocated()
        mem_usage = mem_usage_bytes / 1024 / 1024
        logging.info('mem %d', int(np.ceil(mem_usage)))

        # save best model
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_epoch = epoch
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        scheduler.step()

        # for intermediate feature visualize
        if args.dataset == 'cifar10' and args.visual is True:
            feature_list = visual(visual_idx, visual_data, model)
            feature_dict[epoch] = feature_list

    torch.cuda.synchronize()
    running_time = time.time() - start
    logging.info('search time: {} seconds'.format(running_time))
    if args.dataset == 'cifar10' and args.visual is True:
        torch.save(feature_dict, os.path.join(args.save, 'inter_feature.pt'))

    if 'cls' in args.task:
        logging.info('best arch in epoch {} with valid_acc {}'.format(
            best_epoch, best_metric))
    elif 'rec' in args.task:
        logging.info('best arch in epoch {} with valid_loss {}'.format(
            best_epoch, 1 / best_metric))


def train(train_queue, valid_queue, model, architect: Architect, criterion,
          optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input_, target) in enumerate(train_queue):
        model.train()
        n = input_.size(0)
        input_ = input_.cuda()
        target = target.cuda(non_blocking=True)

        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        mask_search = None
        if args.task == 'rec':
            target = input_.clone()
            target_search = input_search.clone()
        elif args.task == 'mask_rec':
            input_, target, mask_ = utils.mask_imgs(input_, args.patch_size,
                                                    args.mask_ratio)
            input_search, target_search, mask_search = utils.mask_imgs(
                input_search, args.patch_size, args.mask_ratio)
        elif args.task == 'cls_mask':
            cls_target = target
            input_, rec_target, mask_ = utils.mask_imgs(
                input_, args.patch_size, args.mask_ratio)
            target = [cls_target, rec_target]
            cls_target_search = target_search
            input_search, rec_target_search, mask_search = utils.mask_imgs(
                input_search, args.patch_size, args.mask_ratio)
            target_search = [cls_target_search, rec_target_search]

        architect.step(input_,
                       target,
                       input_search,
                       target_search,
                       lr,
                       optimizer,
                       unrolled=args.unrolled,
                       patch_mask=mask_search)

        optimizer.zero_grad()

        logits, _ = model(input_)

        if args.task == 'mask_rec':
            logits = utils.patchify(logits,
                                    patch_size=(args.patch_size,
                                                args.patch_size))
            loss = criterion(logits, target, mask_)
        elif args.task == 'rec' or args.task == 'cls':
            loss = criterion(logits, target)
        elif args.task == 'cls_mask':
            assert type(logits) is list and type(target) is list and type(
                criterion) is list
            l_cls = criterion[0](logits[0], target[0])
            rec_logits = utils.patchify(logits[1],
                                        patch_size=(args.patch_size,
                                                    args.patch_size))
            l_rec = criterion[1](rec_logits, target[1], mask_)

            loss = l_cls + l_rec / (l_rec / l_cls).detach()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        objs.update(loss.data.item(), n)
        total_iter = epoch * len(train_queue) + step
        tb_writer.add_scalar('train/iter_loss', objs.avg, total_iter)

        if 'cls' in args.task:
            if args.task == 'cls_mask':
                logits, target = logits[0], target[0]
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            if 'cls' in args.task:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                             top5.avg)
            elif 'rec' in args.task:
                logging.info('train %03d %e', step, objs.avg)

    return top1.avg, objs.avg


@torch.no_grad()
def infer(valid_queue, model, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input_, target) in enumerate(valid_queue):
        n = input_.size(0)
        input_ = input_.cuda()
        target = target.cuda(non_blocking=True)

        if args.task == 'rec':
            target = input_.clone()
        elif args.task == 'mask_rec':
            input_, target, mask_ = utils.mask_imgs(input_, args.patch_size,
                                                    args.mask_ratio)
        elif args.task == 'cls_mask':
            cls_target = target
            input_, rec_target, mask_ = utils.mask_imgs(
                input_, args.patch_size, args.mask_ratio)
            target = [cls_target, rec_target]

        logits, _ = model(input_)

        # store last batches data every 5 epoch
        if (epoch + 1) % 5 == 0 and samples_saver is not None:
            if 'rec' in args.task:
                samples_saver.update(
                    epoch + 1,
                    target.clone() if args.task == 'rec' else utils.unpatchify(
                        target.clone(), (args.patch_size, args.patch_size)),
                    logits.clone(),
                    None if args.task == 'rec' else input_.clone())
            elif args.task == 'cls_mask':
                samples_saver.update(
                    epoch + 1,
                    utils.unpatchify(rec_target.clone(),
                                     (args.patch_size, args.patch_size)),
                    logits[1].clone(), input_.clone())
            samples_saver.save()

        if args.task == 'mask_rec':
            logits = utils.patchify(logits,
                                    patch_size=(args.patch_size,
                                                args.patch_size))
            loss = criterion(logits, target, mask_)
        elif args.task == 'rec' or args.task == 'cls':
            loss = criterion(logits, target)
        elif args.task == 'cls_mask':
            assert type(logits) is list and type(target) is list and type(
                criterion) is list
            l_cls = criterion[0](logits[0], target[0])
            rec_logits = utils.patchify(logits[1],
                                        patch_size=(args.patch_size,
                                                    args.patch_size))
            l_rec = criterion[1](rec_logits, target[1], mask_)
            loss = l_cls + l_rec / (l_rec / l_cls).detach()

        objs.update(loss.item(), n)

        if 'cls' in args.task:
            if args.task == 'cls_mask':
                logits, target = logits[0], target[0]
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            if 'cls' in args.task:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                             top5.avg)
            elif 'rec' in args.task:
                logging.info('valid %03d %e', step, objs.avg)

    return top1.avg, objs.avg


@torch.no_grad()
def visual(visual_idx, visual_data, model):
    model.eval()
    feature_list = []
    for i in visual_idx:
        img, _ = visual_data[i]
        input_ = img.cuda().unsqueeze(0)  # C,H,W to B,C,H,W
        _, inter_feature = model(input_)
        feature_list.append(inter_feature)

    return feature_list


if __name__ == '__main__':
    main()
