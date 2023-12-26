import math
import sys
from typing import Iterable, Optional

import torch, tqdm

from timm.data import Mixup
from timm.utils import accuracy

import lr_sched
import torch.nn as nn

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    loss_accu = 0.0
    sample_num = 0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in tqdm.tqdm(enumerate(data_loader)):
        sample_num += samples.shape[0]
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs = nn.functional.softmax(outputs, dim=1)
            # outputs = torch.argmax(probabilities, dim=1)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_accu += loss
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

    return loss_accu / sample_num


@torch.no_grad()
def evaluate(data_loader, model, device, prin=False):
    criterion = torch.nn.CrossEntropyLoss()

    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    acc1 = 0
    acc5 = 0
    num = 0
    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if prin:
                print("output")
                print(output)
                print("target")
                print(target)
            loss = criterion(output, target)

        # print(output)
        # # 获取前五个最大值及其对应的索引
        # topk_values1, topk_indices1 = torch.topk(output[0], k=5)
        # topk_values2, topk_indices2 = torch.topk(output[-1], k=5)

        # # 打印前五个最大值及其对应的索引
        # print("Top-5 Values:")
        # print(topk_values1)
        # print(topk_values2)
        # print("Top-5 Indices:")
        # print(topk_indices1)
        # print(topk_indices2)
        # print("targets:")
        # print(target)
        acc1_t, acc5_t = accuracy(output, target, topk=(1, 5))
        num = num + 2
        acc1 = acc1 + acc1_t
        acc5 = acc5 + acc5_t
        batch_size = images.shape[0]

    return {'acc1': acc1/num, 'acc5': acc5/num}