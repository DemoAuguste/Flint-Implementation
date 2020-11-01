from models import *
from flint import *
from utils import *

import argparse
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='lenet5 | mobilenetv2 | resnet18 | resnet34', default='mobilenetv2')
    parser.add_argument('--dataset', '-d', type=str, help='cifar10 | cifar100', default='cifar10')
    parser.add_argument('--version', '-v', type=str, help='version', default='99')
    parser.add_argument('--epoch', '-e', type=int, help='number of epochs', default=200)
    parser.add_argument('--strategy', '-s', type=int, help='strategies 1, 2, and 3.', default=1)
    parser.add_argument('--rate1', '-r1', type=float, help='good sample rate.', default=0.9)
    parser.add_argument('--rate2', '-r2', type=float, help='good sample rate.', default=0.05)

    args = parser.parse_args()

    model = get_model_by_name(args.model, args.dataset)

    indexed_train_loader, test_loader = get_dataloader(args.dataset, indexed=True)
        
    # if args.base:
    #     indexed_train_loader, test_loader = get_dataloader(args.dataset, indexed=False)
    # else:
    #     indexed_train_loader, test_loader = get_dataloader(args.dataset, indexed=True)
    
    model_weights_dir = os.path.join(weights_dir, args.model)
    model_weights_dir = os.path.join(model_weights_dir, args.dataset)
    model_weights_dir = os.path.join(model_weights_dir, args.version)
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train the model.
    flint_train_process(model, indexed_train_loader, test_loader, epochs, optimizer, model_weights_dir)