from .mobilenetv2 import mobilenet_v2
from .resnet import resnet18, resnet34
from .lenet5 import *
from .shufflenetv2 import shufflenet_v2


def get_model_by_name(model_name, dataset_name):
    if dataset_name in ['mnist', 'cifar10', 'svhn']:
        num_classes = 10
    elif dataset_name in ['cifar100']:
        num_classes = 100
    
    if model_name == 'lenet5':
        model = lenet5(num_classes=num_classes, dataset=dataset_name)
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_name == 'shufflenetv2':
        model = shufflenet_v2(num_classes=num_classes)

    return model