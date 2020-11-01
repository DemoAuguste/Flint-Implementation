"""
2020.09.18
原来的idea是把dataset划分出来，按照以往的performance分配不同的learning rate。
实验证明这种方法很难做到repair。

在dataset里面有些sample感觉上是conflicting的，假设有x1和x2，在training的过程中很难保证x1和x2都正确。

现在的想法是首先score training data，实际上这个score也是给dataset的难易度排序了一下。
然后再选出那些很难训练的data，generate一些fake data用来训练。

其实这个思路有点像adversarial training的方法，不同之处在于generated data是随机的。

看了一些data augmentation的方法，感觉这种randomly generated fake data使用有可能提高acc的。

先试试吧。
"""

import torch
import torch.nn as nn
import numpy as np
from settings import batch_size
from utils import progress_bar


def get_perturb_data(input_x, _ord=2, bound=0.1):
    if _ord == 2:
        per_mat = torch.rand(input_x.size())
        per_mat = per_mat / torch.norm(per_mat, p='fro')  # normalize
        per_mat -= 0.5
        per_mat *= 2
        per_mat *= bound
    else:
        raise NotImplementedError('not implemented yet.')

    per_input_x = input_x + per_mat
    return per_input_x

    
def generate_data(dataset, index, num=100, _ord=2, bound=0.1):
    """
    num: number of perturbed data for each data to be generated.
    ord: lp norm.
    bound: the boundary of the perturbation.
    """
    # retrieve data from the dataset. Reform the data.
    input_x = None
    input_y = []
    if isinstance(index, torch.Tensor):
        index = idnex.tolist()
    for idx in index:
        temp_x = dataset.data[idx][0]
        temp_x = temp_x.unsqueeze(0)
        if input_x is None:
            input_x = temp_x
        else:
            input_x = torch.cat((input_x, temp_x))
        input_y.append(int(dataset.data[idx][1]))
    # input_y = torch.tensor(input_y)

    perturb_x = None
    perturb_y = []
    for i in range(num):
        temp_perturb_x = get_perturb_data(input_x, _ord=_ord, bound=bound)
        if perturb_x is None:
            perturb_x = input_x + temp_perturb_x
        else:
            perturb_x = torch.cat((perturb_x, input_x + temp_perturb_x))
        perturb_y.extend(input_y)
        
    perturb_x = perturb_x.clamp(-1.0, 1.0)

    # shuffle the data
    perturb_idx = torch.randperm(perturb_x.size(0))
    perturb_x = perturb_x[perturb_idx]
    perturb_y = torch.tensor(perturb_y)
    # print(perturb_idx.size())
    # print(perturb_y.size())
    # print(perturb_x.size())
    perturb_y = perturb_y[perturb_idx]
    
    return perturb_x, perturb_y


def get_good_data(dataset, index):
    input_x = None
    input_y = []
    print(index.size())
    for idx in index.tolist():
        temp_x = dataset.data[idx][0]
        temp_x = temp_x.unsqueeze(0)
        if input_x is None:
            input_x = temp_x
        else:
            input_x = torch.cat((input_x, temp_x))
        input_y.append(int(dataset.data[idx][1]))
    
    input_idx = torch.randperm(input_x.size(0))
    input_x = input_x[input_idx]
    input_y = torch.tensor(input_y)
    input_y = input_y[input_idx]

    return input_x, input_y


def train_perturbed_one_epoch(model, perturb_x, perturb_y, optimizer, shuffle=True, device='cuda'):

    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    data_num = perturb_x.size(0)
    iter_num, m = divmod(data_num, batch_size)
    if m != 0:
        iter_num += 1

    # shuffle the data again.
    if shuffle:
        perturb_idx = torch.randperm(perturb_x.size(0))
        perturb_x = perturb_x[perturb_idx]
        perturb_y = perturb_y[perturb_idx]
    
    for i in range(iter_num):
        image = perturb_x[batch_size*i:batch_size*(i+1)]
        image = image.to(device)
        target = perturb_y[batch_size*i:batch_size*(i+1)]
        target = target.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        optimizer.step()

        preds = output.data.max(1)[1]

        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(target).sum()

        total += target.size(0)

        progress_bar(i, iter_num, "Train Acc: %.3f%%" % (running_corrects * 100.0 / total))

    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))

    return running_corrects / total