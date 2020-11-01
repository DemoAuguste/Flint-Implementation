import torch
import torch.nn as nn
from .func import progress_bar, AverageMeter
from sklearn.metrics import confusion_matrix


def train_one_epoch(model, train_loader, optimizer, device='cuda'):
    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    for batch_idx, (image, target) in enumerate(train_loader):
        # print(image.size())
        # print(target.size())

        image = image.float()
        image = image.to(device)
        target = target.long()
        target = target.to(device)
        indx_target = target.clone()

        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        optimizer.step()

        preds = output.data.max(1)[1]

        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(indx_target).sum()

        total += target.size(0)

        progress_bar(batch_idx, len(train_loader), "Train Acc: %.3f%%" % (running_corrects * 100.0 / total))

    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))


def indexed_train_one_epoch(model, train_loader, optimizer, device='cuda'):
    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    for batch_idx, (image, target, _) in enumerate(train_loader):
        # print(image.size())
        # print(target.size())

        image = image.float()
        image = image.to(device)
        target = target.long()
        target = target.to(device)
        indx_target = target.clone()

        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        optimizer.step()

        preds = output.data.max(1)[1]

        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(indx_target).sum()

        total += target.size(0)

        progress_bar(batch_idx, len(train_loader), "Train Acc: %.3f%%" % (running_corrects * 100.0 / total))

    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))


def indexed_test(net, test_loader, use_cuda = True):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = net(inputs)

        _, predicted = torch.max(outputs.data, dim=1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        total += targets.size(0)
        progress_bar(batch_idx, len(test_loader), "Test Acc: %.3f%%" % (100.0 * correct / total))

    return correct / total



# test function. use existing implementation.
def test(net, test_loader, use_cuda = True, dataset_name='CIFAR10', n_batches_used=None, eval_model=True):
    """
    A basic test function for forward without additional arguments
    :param net:
    :param test_loader:
    :param use_cuda:
    :param dataset_name:
    :param n_batches_used:
    :return:
    """
    if eval_model:
        net.eval()
    else:
        net.train()

    if dataset_name != 'ImageNet':
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = net(inputs)

            _, predicted = torch.max(outputs.data, dim=1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(test_loader), "Test Acc: %.3f%%" % (100.0 * correct / total))

        return correct / total

    else:
        batch_time = AverageMeter()
        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                losses = nn.CrossEntropyLoss()(outputs, targets)

                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                train_loss.update(losses.data.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 200 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), batch_time=batch_time, loss=train_loss,
                        top1=top1, top5=top5))

                if n_batches_used is not None and batch_idx >= n_batches_used:
                    break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


def get_class_accuracy(model, data_loader, device='cuda', eval_mode=True):
    # model.eval()
    if eval_mode:
        model.eval()
    else:
        model.train()

    predlist=torch.zeros(0,dtype=torch.long, device=device)
    lbllist=torch.zeros(0,dtype=torch.long, device=device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, dim=1)

            predlist=torch.cat([predlist, preds.view(-1)])
            lbllist=torch.cat([lbllist, targets.view(-1)])

    conf_mat=confusion_matrix(lbllist.data.cpu().numpy(), predlist.data.cpu().numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print(class_accuracy)

    return conf_mat
    