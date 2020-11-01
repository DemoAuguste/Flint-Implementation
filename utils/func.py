import sys
import time
import torch

from sklearn.metrics import confusion_matrix


term_width = 100

TOTAL_BAR_LENGTH = 15.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)


def divide_dataset_by_class(data_loader, label):
    dataset = data_loader.dataset
    idx = torch.tensor(dataset.targets) == label
    class_data = torch.tensor(dataset.data[idx])
    class_data = class_data.permute(0, 3, 1, 2) # reorder the axis.
    class_targets = torch.ones(class_data.shape[0]) * label

    # change the dtype
    class_data = class_data.type(torch.float32)
    class_targets = class_targets.type(torch.int64)

    return (class_data, class_targets) # tensor


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
    # class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    # print(class_accuracy)

    return conf_mat