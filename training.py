import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch
from matplotlib import pyplot as plt
import datetime
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from math import cos, pi
from sklearn.model_selection import train_test_split
from ATCN import ATCNConv3d
import os

parser = argparse.ArgumentParser(description='scTGRN Training')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
# parser.add_argument('--num_classes', default=2, type=int, help='num of classes')
# parser.add_argument('--data_path', default="/home/scTGRN/mesc1_4D_input_tensors", type=str, help='training directory')
parser.add_argument('--model_name', default="scRNA_data", type=str, help='model name')
args = parser.parse_args()

def load_data(indel_list,data_path):
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:
        xdata = np.load(data_path+'/NTxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(int(len(ydata)/2)):
            xxdata_list.append(xdata[2*k,:,:,:,:])
            xxdata_list.append(xdata[2*k+1,:,:,:,:])
            yydata.append(1)
            yydata.append(0)
        count_setx = count_setx + int(len(yydata))
        count_set.append(count_setx)
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    return((np.array(xxdata_list),yydata_x,count_set))

# adjust the learning rate
def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0.00001, lr_max=0.01, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch-max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_acc(predictions, labels):
    return torch.sum(predictions == labels)/(labels.shape[0]*labels.shape[1])

def get_recall(predictions, labels):
    return torch.sum((predictions == labels) * (labels == 1))/torch.sum(labels == 1)

def get_precision(predictions, labels):
    return torch.sum((predictions == labels) * (labels == 1))/torch.sum(predictions == 1)

##############
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
batch_size = 256
num_classes = 2
epoches = 200
lr_max = 0.01
lr_min = 0.00001
max_epoch = 10*5
lrs = []
length_TF = 36
whole_data_TF = [i for i in range(length_TF)]
data_path = '/home/scTGRN/mesc1_4D_input_tensors'
file = open(r"./mesc1_result.txt", "w", encoding='utf-8')


#############
# randomly split the dataset
data_TF = whole_data_TF
(x_data, y_data, count_set_data) = load_data(data_TF, data_path)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, shuffle=True, stratify=y_data)
train_data, train_label, test_data, test_label = x_train, y_train, x_test, y_test
print(x_train.shape, 'x_train samples', type(x_train))
print(x_test.shape, 'x_test samples', type(x_test))
print(y_train.shape, 'y_train samples', type(y_train))
print(y_test.shape, 'y_test samples', type(y_test))

##############
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
best_acc, best_auc, best_aupr, best_f1, best_fpr, best_tpr, start_epoch = 0, 0, 0, 0, 0, 0, 0
print('==> Building model..')
net = ATCNConv3d(1, 1, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64], kernel_size=(3, 3, 3), space_dilation=1, groups=1, dropout=0.05, activation=nn.ReLU)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

# acc curve and loss curve
train_losses = []
train_acces = []
test_losses = []
test_acces = []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total, train_acc = 0, 0, 0, 0
    def get_batch(batch_size, i):
        x = batch_size * i
        train_data_batch = train_data[x:x + batch_size, :]
        train_lable_batch = train_label[x:x + batch_size]
        return train_data_batch, train_lable_batch
    shape_t = train_data.shape
    num_train_data = shape_t[0]
    batch_num = int(num_train_data // batch_size)
    adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=max_epoch, lr_min=lr_min,lr_max=lr_max, warmup=True)
    print(optimizer.param_groups[0]['lr'])
    lrs.append(optimizer.param_groups[0]['lr'])

    for i in range(batch_num):
        inputs, targets = get_batch(batch_size, i)
        inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        predicted = torch.where(outputs > 0, 1, 0)
        predicted = np.squeeze(predicted)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        scores = outputs
        scores = np.squeeze(scores)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ACC = metrics.accuracy_score(targets.cpu(), predicted.cpu())
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | ACC11: %.3f%%'
              % (epoch + 1, i + 1, loss.item(), 100. * correct / total, ACC1))
        acc = correct / total
        train_acc += acc
    train_losses.append(train_loss / batch_num)
    train_acces.append(100 * train_acc / batch_num)
    return loss.item(), ACC

def test(epoch):
    global best_acc, best_auc, best_aupr, best_f1, best_fpr, best_tpr
    net.eval()
    test_loss, correct, total, test_acc = 0, 0, 0, 0
    pred_label = []
    prob_all = []
    label_all = []
    def get_batch(batch_size, i):
        x = batch_size * i
        test_data_batch = test_data[x:x + batch_size, :]
        test_lable_batch = test_label[x:x + batch_size]
        return test_data_batch, test_lable_batch

    shape_t = test_data.shape
    num_train_data = shape_t[0]
    batch_num = int(num_train_data // batch_size)
    with torch.no_grad():
        for i in range(batch_num):
            inputs, targets = get_batch(batch_size, i)
            inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            predicted = torch.where(outputs > 0, 1, 0)
            predicted = np.squeeze(predicted)
            loss = criterion(outputs, targets.unsqueeze(1))
            test_loss += loss.item()
            scores = outputs
            scores = np.squeeze(scores)
            pred_label.extend(predicted.cpu().numpy())
            prob_all.extend(scores.cpu().numpy())
            label_all.extend(targets.to(device).cpu())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total
            test_acc += acc
        label_all = np.array(label_all)
        prob_all = np.array(prob_all)
        fpr, tpr, thresholds = metrics.roc_curve(label_all, prob_all, pos_label=1)
        auc = np.trapz(tpr, fpr)
        precesion_aupr, recall_aupr, _ = metrics.precision_recall_curve(label_all, prob_all)
        aupr = metrics.auc(recall_aupr, precesion_aupr)
        f1 = metrics.f1_score(label_all, pred_label)
        test_losses.append(test_loss / batch_num)
        test_acces.append(100 * test_acc / batch_num)
    acc = metrics.accuracy_score(label_all, pred_label)
    print("Test ACC:", acc)
    print("AUC:", auc)
    print("AUPR:", aupr)
    print("F1:", f1)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('trained_models'):
            os.mkdir('trained_models')
        torch.save(state, r'trained_models\\' + args.model_name)
        best_acc = acc
    if auc > best_auc:
        best_auc = auc
        best_fpr = fpr
        best_tpr = tpr
    if aupr > best_aupr:
        best_aupr = aupr
    if f1 > best_f1:
        best_f1 = f1
    return acc, auc, aupr, f1

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# start = datetime.datetime.now()
if __name__ == '__main__':
    for epoch in range(epoches):
        train_loss, train_acc = train(epoch)
        print('------------------------------')
        test_acc, test_auc, test_aupr, test_f1 = test(epoch)
        file.write("epoch:{}, train_loss:{}, train_acc:{}\n".format(epoch + 1, train_loss, train_acc))
        file.write("epoch:{}, test_acc:{}, test_auc:{}, test_aupr:{}, test_f1:{}\n".format(epoch + 1, test_acc, test_auc, test_aupr, test_f1))
    file.write("Best Test ACC:{}, Best Test AUC:{}, Best Test AUPR:{}, Best Test F1:{}\n".format(best_acc, best_auc, best_aupr, best_f1))
    loss_name = 'loss.png'
    acc_name = 'acc.png'
    loss_title = 'train and test loss'
    acc_title = 'train and test accuracy'
    epo = range(1, epoches + 1)
    plt.figure(0)
    plt.plot(epo, smooth_curve(train_acces), label='Train ACC')
    plt.plot(epo, smooth_curve(test_acces), label='Test ACC')
    plt.title(acc_title)
    plt.legend(loc='lower right')
    plt.savefig(acc_name, dpi=1000)
    plt.figure(1)
    plt.plot(epo, smooth_curve(train_losses), label='Train Loss')
    plt.plot(epo, smooth_curve(test_losses), label='Test Loss')
    plt.title(loss_title)
    plt.legend(loc='upper right')
    plt.savefig(loss_name, dpi=1000)
    auc_name = 'auc.png'
    plt.figure(figsize=(10, 10))
    plt.plot(best_fpr, best_tpr)
    plt.grid()
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title('AUC:'.format(best_auc))
    plt.savefig(auc_name, dpi=1000)
file.close()
