import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

from utils import progress_bar
from settings import batch_size

from .custom_sampler import CustomSampler

PI = 3.1415926
base_lr = 0.0005
original_lr = 0.01


def cosine(input_x, a=1, b=1, c=1):
    # y = a*cos(bx) + c
    if not isinstance(input_x, torch.Tensor):
        input_x = torch.tensor(input_x)
    return a * torch.cos(b * input_x) + c

def sine(input_x, a=1, b=1, c=1):
    # y = a*cos(bx) + c
    if not isinstance(input_x, torch.Tensor):
        input_x = torch.tensor(input_x)
    return a * torch.cos(b * input_x) + c



def adjust_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr']  = learning_rate


def update_pred_matrix(model, epoch, indexed_data_loader, pred_matrix, device='cuda'):
    """
    the data_loader loads the data from a indexed dataset.
    """
    
    for batch_idx, (image, target, index) in enumerate(indexed_data_loader):
        image = image.float()
        image = image.to(device)
        target = target.long()
        target = target.to(device)
        indx_target = target.clone()

        output = model(image)
        preds = output.data.max(1)[1]
        preds = preds.eq(indx_target)
        pred_matrix[index, epoch] = preds.cpu().data

    return pred_matrix


def rose_train_one_epoch(model, epoch, indexed_train_loader, optimizer, pred_matrix, device='cuda'):
    """
    pred_matrix: save the prediciton results in each epoch. 0 represents incorrect and 1 represents correct.
    epoch: the i-th epoch.
    """

    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    for batch_idx, (image, target, index) in enumerate(indexed_train_loader):

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

        progress_bar(batch_idx, len(indexed_train_loader), "Train Acc: %.3f%%" % (running_corrects * 100.0 / total))

    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))

    # update pred_matrix here.
    if epoch != 0:
        # add a new column
        col = torch.zeros((len(indexed_train_loader.dataset), 1), dtype=torch.bool)
        pred_matrix = torch.cat((pred_matrix, col), dim=1)

    pred_matrix = update_pred_matrix(model, epoch, indexed_train_loader, pred_matrix)

    return pred_matrix, running_corrects / total



def rose_train_one_epoch_enhanced(model, epoch, indexed_train_loader, optimizer, indicator_mat, pred_mat, device='cuda'):
    """
    indicator_mat: indicate if the sample is classified correctly.
    pred_mat: save the classification probability for each sample.
    epoch: the i-th epoch.
    """
    # normal training process.
    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    if epoch != 0:
        # add a new column
        ind_col = torch.zeros((len(indexed_train_loader.dataset), 1), dtype=torch.bool)
        pred_col = torch.zeros((len(indexed_train_loader.dataset), 1), dtype=torch.float)
        indicator_mat = torch.cat((indicator_mat, ind_col), dim=1)
        pred_mat = torch.cat((pred_mat, pred_col), dim=1)


    for batch_idx, (image, target, index) in enumerate(indexed_train_loader):

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

        # update matrices.
        # preds = preds.eq(indx_target)
        indicator_mat[index, epoch] = preds.eq(indx_target).cpu().data

        softmax = nn.Softmax(dim=1)
        soft_output = softmax(output)
        pred_mat[index, epoch] = soft_output.data.max(1)[0].cpu().data


        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(indx_target).sum()

        total += target.size(0)

        progress_bar(batch_idx, len(indexed_train_loader), "Train Acc: %.3f%%" % (running_corrects * 100.0 / total))

    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))

    # update indicator_mat and pred_mat.
    

    # # update pred_matrix here.
    # if epoch != 0:
    #     # add a new column
    #     col = torch.zeros((len(indexed_train_loader.dataset), 1), dtype=torch.bool)
    #     pred_matrix = torch.cat((pred_matrix, col), dim=1)

    # pred_matrix = update_pred_matrix(model, epoch, indexed_train_loader, pred_matrix)

    return indicator_mat, pred_mat, running_corrects / total



# investigate the correctness of the data sample.
# this function ranks the data sample according to their predicition results in the latest window_size training steps.
# consider different ranking functions.
def investigate_samples(pred_matrix, window_size=20):
    # get the latest window_size results.
    sub_pred_matrix = pred_matrix[:, -window_size:]
    print(sub_pred_matrix.size())
    last_pred = pred_matrix[:, -1]

    def get_ranking_score(preds):
        # return the score of a pred list.
        score_list = torch.zeros(preds.size())
        for i in range(int(window_size)):
            score_list[i] = sine(i/window_size, a=1, b=PI/2, c=1)
        # for i in range(int(window_size / 2)):
        #     score_list[i] = i * 1 / (window_size / 2)
        # score_list[window_size:] = 1
        score = score_list * preds
        return score.sum()

    score_mat = torch.zeros(pred_matrix.size(0))
    for i in range(sub_pred_matrix.size(0)):
        score_mat[i] = get_ranking_score(sub_pred_matrix[i])

    sorted_score, sorted_indx = torch.sort(score_mat, descending=True)

    print(sorted_score)
    print(sorted_indx)

    # find the index of the max value.
    sub_pred_sum = sub_pred_matrix.sum(dim=1)
    indx = sub_pred_sum == window_size
    max_socre_num = indx.sum()
    # max_socre_index = pred_matrix.size(0)  # last index of max value.
    print(max_socre_num)
    max_socre_index = max_socre_num

    return sorted_indx, max_socre_index

    # region NOT used
    # print(last_pred.sum())
    
    # # sub_pred_sum: calcuate the case of each sample.
    # # there are several cases:
    # #   * window_size: correct prediciton in the last window_size training steps.
    # #   * < window_size and > 0: sometimes correct. Needs to be fixed.
    # #   * 0: incorrect in all training steps. Needs to be fixed.
    # # case 1: 0, case 2: 1, case 3: 2.
    # sub_pred_sum = sub_pred_matrix.sum(dim=1)  # sum by row.

    # case_indicator = torch.zeros((last_pred.size()))
    # print(case_indicator.size())
    # # case 1
    # indx = sub_pred_sum == window_size
    # print(indx.sum())
    # # case_indicator[indx] = 0

    # # case 2 and case 3
    # indx = sub_pred_sum < window_size
    # print(indx.sum())
    # case_indicator += indx

    # # case 3
    # indx = sub_pred_sum == 0
    # print(indx.sum())
    # case_indicator += indx

    # print(case_indicator)
    # return case_indicator
    # endregion

    
def rose_adaptive_train_one_epoch(model, epoch, indexed_train_loader, optimizer, pred_matrix, window_size=20, device='cuda'):

    model.train()

    dataset = indexed_train_loader.dataset
    # investigate the data. Index of the training dataset.
    sorted_index, max_score_index = investigate_samples(pred_matrix, window_size=window_size)

    # max_score_index = int(len(sorted_index) / 2) # test
    # max_score_index = int(len(sorted_index)/2) 

    # adjust the batch accordingly.
    # design a batch evaluation metric.
    cus_sampler_1 = CustomSampler(sorted_index[max_score_index:], batch_size=batch_size, shuffle=False)
    # SeqSampler_1 = BatchSampler(sampler=cus_sampler_1, 
    #                             batch_size=batch_size, 
    #                             drop_last=False)
    fault_dataloader = DataLoader(indexed_train_loader.dataset, batch_size=batch_size, shuffle=False, sampler=cus_sampler_1)
    
    cus_sampler_2 = CustomSampler(sorted_index[:max_score_index], batch_size=batch_size, shuffle=True)
    # SeqSampler_2 = BatchSampler(sampler=cus_sampler_2, 
    #                             batch_size=batch_size, 
    #                             drop_last=False)
    correct_dataloader = DataLoader(indexed_train_loader.dataset, batch_size=batch_size, shuffle=False, sampler=cus_sampler_2)


    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    model = model.to(device)


    print(len(fault_dataloader))
    # # the first loop.
    for batch_idx, (image, target, index) in enumerate(fault_dataloader):
        # drop last.
        if batch_idx == len(fault_dataloader) - 1:
            break
        image = image.float()
        image = image.to(device)
        target = target.long()
        target = target.to(device)
        indx_target = target.clone()

        # set the learning rate.
        lr = cosine(batch_idx / len(fault_dataloader) - 1, a=original_lr, b=PI, c=original_lr + base_lr)
        # lr = 0.02
        adjust_learning_rate(optimizer, learning_rate=lr)

        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        optimizer.step()

        preds = output.data.max(1)[1]

        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(indx_target).sum()

        total += target.size(0)

        progress_bar(batch_idx, len(fault_dataloader), "Train Acc: %.3f%%, lr= %.5f%%" % (running_corrects * 100.0 / total, lr))
        


    # the second loop.
    print(len(correct_dataloader))
    for batch_idx, (image, target, index) in enumerate(correct_dataloader):
        # drop last.
        if batch_idx == len(correct_dataloader) - 1:
            break
        image = image.float()
        image = image.to(device)
        target = target.long()
        target = target.to(device)
        indx_target = target.clone()

        # set the learning rate.
        lr = 0.005
        adjust_learning_rate(optimizer, learning_rate=lr)

        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        optimizer.step()

        preds = output.data.max(1)[1]

        running_loss += loss.item() * image.size(0)
        running_corrects += preds.eq(indx_target).sum()

        total += target.size(0)

        progress_bar(batch_idx, len(correct_dataloader), "Train Acc: %.3f%%, lr= %.5f%%" % (running_corrects * 100.0 / total, lr))


    print('loss: {:.4f}, training acc: {:.4f}'.format(running_loss, running_corrects * 100.0 / total))

    # adjust_learning_rate(optimizer, learning_rate=original_lr)

    # update pred_matrix here.
    if epoch != 0:
        # add a new column
        col = torch.zeros((len(indexed_train_loader.dataset), 1), dtype=torch.bool)
        pred_matrix = torch.cat((pred_matrix, col), dim=1)

    pred_matrix = update_pred_matrix(model, epoch, indexed_train_loader, pred_matrix)

    return pred_matrix, running_corrects / total
    
    
def get_data_by_index(selected_index, dataset, shuffle=True):
    sampler = CustomSampler(selected_index, batch_size=batch_size, shuffle=shuffle)
    sample_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    sample_x = None
    sample_y = None
    for batch_x, batch_y, _ in sample_dataloader:
        if sample_x is None:
            sample_x = batch_x
            sample_y = batch_y.unsqueeze(1)
        else:
            sample_x = torch.cat((sample_x, batch_x))
            sample_y = torch.cat((sample_y, batch_y.unsqueeze(1)))
    sample_y = sample_y.squeeze()

    return sample_x, sample_y

