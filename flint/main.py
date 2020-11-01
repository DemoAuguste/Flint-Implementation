import torch
from torch.utils.data import DataLoader, Dataset
import os

from utils import train_one_epoch, indexed_train_one_epoch, test, Logger, indexed_test
from .func import *
from .ada import *

import random


def flint_train_process(model, indexed_train_loader, test_loader, epochs, optimizer, model_weights_dir):
    # initialize the pred_matrix.
    data_num = len(indexed_train_loader.dataset)
    indicator_mat = torch.zeros([data_num, 1], dtype=torch.bool)
    pred_mat = torch.zeros([data_num, 1], dtype=torch.float)

    print(pred_mat.size())

    train_acc_seq = []
    test_acc_seq = []
    train_acc_file_name = os.path.join(model_weights_dir, 'base_train_acc.pth')
    test_acc_file_name = os.path.join(model_weights_dir, 'base_test_acc.pth')

    best_weights_path = os.path.join(model_weights_dir, 'best_weights.pth')

    highest_test_acc = 0.0
    # initialization training.
    for epoch in range(epochs): # test
        print('[epoch {}]'.format(epoch))
        indicator_mat, pred_mat, train_acc = rose_train_one_epoch_enhanced(model, epoch, indexed_train_loader, optimizer, indicator_mat, pred_mat)
        test_acc = test(model, test_loader)
        
        train_acc_seq.append(train_acc)
        test_acc_seq.append(test_acc)
        torch.save(torch.tensor(train_acc_seq), train_acc_file_name)
        torch.save(torch.tensor(test_acc_seq), test_acc_file_name)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_weights_dir, 'checkpoint-regular-{}.pth'.format(epoch + 1)))
            torch.save(indicator_mat, os.path.join(model_weights_dir, 'indicator-matrix-{}.pth'.format(epoch + 1)))
            torch.save(pred_mat, os.path.join(model_weights_dir, 'prediction-matrix-{}.pth'.format(epoch + 1)))

        if epoch > 60 and test_acc > highest_test_acc:
            torch.save(model.state_dict(), best_weights_path)
            highest_test_acc = test_acc


def flint_fix_process(model, indexed_train_loader, test_loader, epochs, optimizer, model_weights_dir, strategy=1, rate=0.01, good_rate=0.999, reverse=False, generate_num=10, file_name=None):
    if file_name is None:
        logger = Logger(model_weights_dir, 'enhanced-{}.txt'.format(strategy))
    else:
        logger = Logger(model_weights_dir, file_name)
        
    logger.log('---------------------------- start ----------------------------')
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed_enhanced-{}.pth'.format(strategy))

    logger.log('PARAMETERS:')
    logger.log('strategy: {}, rate: {}, good rate: {}, generate num: {}'.format(strategy, rate, good_rate, generate_num))

    model.load_state_dict(torch.load(os.path.join(model_weights_dir, 'checkpoint-regular-{}.pth'.format(epochs))))
    model.to('cuda')
    original_train_acc = indexed_test(model, indexed_train_loader)
    original_test_acc = test(model, test_loader)
    logger.log('original training accuracy: {:.4f}'.format(original_train_acc))
    logger.log('original test accuracy: {:.4f}'.format(original_test_acc))


    indicator_mat = torch.load(os.path.join(model_weights_dir, 'indicator-matrix-{}.pth'.format(epochs)))
    prediction_mat = torch.load(os.path.join(model_weights_dir, 'prediction-matrix-{}.pth'.format(epochs)))

    indicator_sum = indicator_mat.sum(dim=1)
    pred_sum = prediction_mat.sum(dim=1)
    pred_sum = pred_sum / prediction_mat.size(1)

    sort_list = [(i, j, k) for i,j, k in zip(indicator_sum.numpy(), pred_sum.numpy(), [q for q in range(indicator_mat.size(0))])]
    sort_list.sort(key=lambda x: (x[0], x[1]), reverse=True)  # from high quality to low quality. format: [indicator, pred prob, index]

    if reverse:
        good_list = sort_list[-int(prediction_mat.size(0) * good_rate):-1]
    else:
        good_list = sort_list[:int(prediction_mat.size(0) * good_rate)]
    print(len(sort_list))
    print(good_rate)
    print(prediction_mat.size())
    print(len(good_list))
    good_num = len(good_list)
    
    # select samples to mutate.
    # three strategies:
    #  - 1: select the top samples.
    #  - 2: select the bottom samples.
    #  - 3: select the data randomly.
    selected_num = int(prediction_mat.size(0) * rate)
    if strategy == 1:
        selected_sample = good_list[:selected_num]
    elif strategy == 2:
        selected_sample = sort_list[-selected_num:-1]
    elif strategy == 3:
        # sliced_sort_list = sort_list[:len(sort_list) * ]
        random.shuffle(good_list)
        selected_sample = good_list[:selected_num]
    elif strategy == 4:
        # identify the middle sample.
        selected_sample = good_list[int(good_num * 0.2):int(good_num * 0.2) + selected_num]
        

    # retrieve the index of the sample.
    # select sample to be perturbed.
    selected_index = [item[-1] for item in selected_sample]
    # sample_x, sample_y = get_data_by_index(selected_index, indexed_train_loader.dataset, shuffle=True)

    # get the good samples.
    good_index = [item[-1] for item in good_list]
    good_x, good_y = get_data_by_index(good_index, indexed_train_loader.dataset, shuffle=True)

    highest_train_acc = original_train_acc
    highest_test_acc = original_test_acc

    for _ in range(2):  # test
        perturb_x, perturb_y = generate_data(indexed_train_loader.dataset, selected_index, num=generate_num, _ord=2, bound=0.01)
        print('concatenate the data...')
        syn_x = torch.cat((good_x, perturb_x))
        syn_y = good_y.tolist()
        syn_y.extend(perturb_y.tolist())
        syn_y = torch.tensor(syn_y)

        # shuffle the data.
        syn_idx = torch.randperm(syn_x.size(0))
        syn_x = syn_x[syn_idx]
        syn_y = syn_y[syn_idx]
        # print(perturb_idx.size())
        # print(syn_y.size())
        # print(syn_x.size())

        for _ in range(3):
            train_perturbed_one_epoch(model, syn_x, syn_y, optimizer)
        temp_test_acc = test(model, test_loader)

        if temp_test_acc > highest_test_acc:
            logger.log('test accuracy improved from {:.4f} to {:.4f}'.format(highest_test_acc, temp_test_acc))
            highest_test_acc = temp_test_acc
            torch.save(model.state_dict(), fixed_weights_path)

            # add a normal training process.
            # indexed_train_one_epoch(model, indexed_train_loader, optimizer)

            # get the training accuracy.
            temp_train_acc = indexed_test(model, indexed_train_loader)
            # temp_test_acc = test(model, test_loader)
            highest_train_acc = temp_train_acc

            logger.log('training accracy after the improvement: {:.4f}'.format(temp_train_acc))
            logger.log('test accracy after the improvement: {:.4f}'.format(temp_test_acc))

    logger.log('---------------------------- end ----------------------------\n\n')

    return highest_train_acc, highest_test_acc

