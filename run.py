import argparse
import os
import shutil
import time
import random
import numpy as np

import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
from models import *
# os.rename('pytorch-classification', 'pytorch_classification')
import pytorch_classification
from pytorch_classification.models.cifar.resnet import *
from pytorch_classification.models.cifar.preresnet import *
from pytorch_classification.models.cifar.densenet import *

from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=64, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=None,
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--plateauLR', type=int, default=None, help='Reduce LR on plateu patience')
parser.add_argument('--unfreeze', type=int, default=1,help='Epoch at which to unfreeze body')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument("--wandb_name", default='test_project', help="Name of project in wandb")
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--active_save', default=None, type=int, help='Specify at what intervals to save to wandb')
# Architecture
parser.add_argument("-arch","--arch", help="Base model which is modefied into a Hydra type network")
parser.add_argument('--split_pt', type=int, help='Point in network where it splits up into heads')
parser.add_argument('--n_heads', type=int, help='Number of heads')
parser.add_argument('--pretrained_body', action='store_false',help='pretrained weights of base model')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
wandb.init(project=args.wandb_name, name=args.arch+"_"+str(args.n_heads)+"_"+str(args.split_pt), config=args)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
model = None

def main():
    global best_acc
    global model
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, drop_last=True, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, drop_last=True, shuffle=False, num_workers=args.workers)
    body_path = None
    if (args.pretrained_body == True):
      model_name = ""
      if (args.arch == 'densenet'):
        model_name = "densenetbc_100_12_"
      elif (args.arch == 'preresnet'):
        model_name = "preresnet110_"
      elif (args.arch == 'resnet'):
        model_name = "resnet110_"
      else:
        raise ValueError("Base model architecture not recognised")
      body_path = "pretrained models/"+model_name+args.dataset+"/model_best.pth.tar"

    # Model
    print("==> creating model '{}'".format(args.arch))
    model = HydraNet(args.arch, n_heads=args.n_heads, split_pt=args.split_pt, num_classes=num_classes, batch_size=args.train_batch, path=body_path)
    sample_wts = model.sample_wts
    wandb.watch(model, log='all')
    model = torch.nn.DataParallel(model).cuda()
    freeze_body(model)
    cudnn.benchmark = True

    n_params = sum(p.numel() for p in model.parameters())/1000000.0
    print('    Total params: %.2fM' % (n_params))
    wandb.config.update({'No. of parameters': n_params})

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        sample_wts = checkpoint['sample_wts']
        model._modules['module'].sample_wts = checkpoint['sample_wts']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    if args.plateauLR!=None:
      scheduler = ReduceLROnPlateau(optimizer, patience=args.plateauLR)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # lr scheduler and unfreeze body scheduler
        if args.plateauLR==None:
          adjust_learning_rate(optimizer, epoch+1)
        if (epoch+1)==args.unfreeze:
          freeze_body(model, False)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, sample_wts, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # reduce lr on plateau if specified
        if args.plateauLR!=None:
          scheduler.step(train_loss)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        wandb.run.summary["best_acc"] = best_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'sample_wts' : sample_wts
            }, is_best, checkpoint=args.checkpoint)
        if (args.active_save != None) and (epoch%args.active_save == 0):
          wandb.save('checkpoint')

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    wandb.save('checkpoint')

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, sample_wts, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.n_heads)]
    losses_avg = AverageMeter()
    top1 = [AverageMeter() for i in range(args.n_heads)]
    top5 = [AverageMeter() for i in range(args.n_heads)]
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
#         print('.', end='')
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        optimizer.zero_grad()
        for head_idx in range(args.n_heads):
          loss = criterion(outputs[head_idx], targets)
          loss = (loss * sample_wts[head_idx] / sample_wts[head_idx].sum()).sum()

          # measure accuracy and record loss
          prec1, prec5 = accuracy(outputs[head_idx].data, targets.data, topk=(1, 5))
          if float(torch.__version__[:3]) < 0.5:
              losses[head_idx].update(loss.data[0], inputs.size(0))
              top1[head_idx].update(prec1[0], inputs.size(0))
              top5[head_idx].update(prec5[0], inputs.size(0))
          else:
              losses[head_idx].update(loss.data, inputs.size(0))
              top1[head_idx].update(prec1, inputs.size(0))
              top5[head_idx].update(prec5, inputs.size(0))

          # compute gradient and do SGD step
          loss.backward(retain_graph=True)
        losses_avg.update(sum([h.avg for h in losses])/len(losses), inputs.size(0))
        top1_avg.update(sum([h.avg for h in top1])/len(top1), inputs.size(0))
        top5_avg.update(sum([h.avg for h in top5])/len(top5), inputs.size(0))
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_avg: {loss:.4f} | top1_avg: {top1: .4f} | top5_avg: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses_avg.avg,
                    top1=top1_avg.avg,
                    top5=top5_avg.avg,
                    )
        bar.next()
    bar.finish()
    wandb.log({"top1": [h.avg for h in top1], 
               "top1_avg": top1_avg.avg,
               "top5": [h.avg for h in top5], 
               "top5_avg": top5_avg.avg,
               "losses": [h.avg for h in losses],
               "losses_avg": losses_avg.avg}, step=epoch)
    return (losses_avg.avg, top1_avg.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          # measure data loading time
          data_time.update(time.time() - end)

          if use_cuda:
              inputs, targets = inputs.cuda(), targets.cuda()

          # compute output
          outputs = model(inputs)
          outputs_sum = outputs[0].cuda()
          for i in range(1, args.n_heads):
            outputs_sum = torch.add(outputs_sum, outputs[i].cuda())
          outputs = outputs_sum/args.n_heads
          loss = criterion(outputs, targets)
          loss = torch.mean(loss)

          # measure accuracy and record loss
          prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
          if float(torch.__version__[:3]) < 0.5:
              losses.update(loss.data[0], inputs.size(0))
              top1.update(prec1[0], inputs.size(0))
              top5.update(prec5[0], inputs.size(0))
          else:
              losses.update(loss.data, inputs.size(0))
              top1.update(prec1, inputs.size(0))
              top5.update(prec5, inputs.size(0))

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()

          # plot progress
          bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                      batch=batch_idx + 1,
                      size=len(testloader),
                      data=data_time.avg,
                      bt=batch_time.avg,
                      total=bar.elapsed_td,
                      eta=bar.eta_td,
                      loss=losses.avg,
                      top1=top1.avg,
                      top5=top5.avg,
                      )
          bar.next()
    bar.finish()
    wandb.log({"top1 test": top1.avg, 
               "top5 test": top5.avg,
               "losses test": losses.avg}, step=epoch)
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()