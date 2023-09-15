import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import csv
import os
import numpy as np
import Ndata
import OOD_Metric

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--data', default="iris", type=str, help='dataset')
parser.add_argument('--alg', default="baseline", type=str, help='alg')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--epoch', default=10, type=int, help='total epochs to run')
parser.add_argument('--rho', default=0.2, type=float, help='rho')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
parser.add_argument('--normalization', action='store_true', default=False)
args = parser.parse_args()

class CCLoss(nn.Module):
    """docstring for CCLoss"""
    def __init__(self, alpha):
        super(CCLoss, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()
        self.NLLoss = NLLoss()
        self.alpha = alpha
        self.eps = 1e-5

    def forward(self, outputs, labels):
        num_classes = outputs.size(1)

        SoftmaxOutputs = self.Softmax(outputs)
        confidence = SoftmaxOutputs.max(dim = 1)[0].data
        # confidence = torch.gather(SoftmaxOutputs, dim=1, index=labels.unsqueeze(1)).squeeze(1).data

        CE = - torch.gather(self.LogSoftmax(outputs), dim=1, index=labels.unsqueeze(1)).squeeze(1)
        res = - torch.log(1 - SoftmaxOutputs + self.eps)
        NL = res.sum(dim = 1) - torch.gather(res, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        # NL = NL - res[:, num_classes]

        loss_CE = CE.mean()
        loss_NL = NL.mean()
        loss = ((1 - self.alpha) * confidence * CE  + self.alpha * (1 - confidence) * NL).mean()

        return loss

#Negative Learning
class NLLoss(nn.Module):
    """docstring for CRLoss"""
    def __init__(self):
        super(NLLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.Softmax = nn.Softmax(dim=1)
        self.eps = 1e-5
    def forward(self, outputs, labels):
        num_classes = outputs.size(1)
        res = - torch.log(1 - self.Softmax(outputs) + self.eps)
        ALL = res.sum(dim = 1)
        GT = torch.gather(res, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        loss = (ALL - GT).mean()
        return loss

# Self-Paced Learning
class SPLoss(nn.Module):
    """docstring for SPLoss"""
    def __init__(self):
        super(SPLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()
        self.lam = 2
    def forward(self, outputs, labels):
        loss_values = - torch.gather(self.LogSoftmax(outputs), dim=1, index=labels.unsqueeze(1)).squeeze(1).data
        threshold = loss_values.mean() - 0.01
        mask = torch.le(loss_values, threshold)
        num = mask.sum().item()

        loss_zero = torch.FloatTensor([0.0])
        loss = loss_zero if num == 0 else self.CELoss(outputs[mask], labels[mask])
        return loss

# bootstrapping
class BTLoss(nn.Module):
    """docstring for BTLoss"""
    def __init__(self):
        super(BTLoss, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.beta = 0.5
    def forward(self, outputs, labels):
        num_classes = outputs.size(1)
        targets_t = torch.eye(num_classes)[labels,:].cuda()
        targets_q = self.Softmax(outputs).data
        targets = self.beta * targets_t + (1 - self.beta) * targets_q
        loss = - (targets * self.LogSoftmax(outputs)).sum(dim=1).mean()
        return loss

# Symmetric Cross Entropy
class SCELoss(nn.Module):
    """docstring for SCELoss"""
    def __init__(self):
        super(SCELoss, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()
    def forward(self, outputs, labels):
        num_classes = outputs.size(1)
        targets = torch.eye(num_classes)[labels,:].cuda()
        loss_1 = - (self.Softmax(targets) * self.LogSoftmax(outputs)).sum(dim=1).mean()
        loss_2 = - (self.Softmax(outputs) * self.LogSoftmax(targets)).sum(dim=1).mean()
        loss = loss_1 + loss_2
        return loss

# MAE
class MAELoss(nn.Module):
    """docstring for MAELoss"""
    def __init__(self):
        super(MAELoss, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.L1Loss = nn.SmoothL1Loss()
    def forward(self, outputs, labels):
        loss = self.L1Loss(torch.gather(self.Softmax(outputs), dim=1, index=labels.unsqueeze(1)).squeeze(1), torch.ones(outputs.size(0)).cuda())
        return loss

class ILLoss(nn.Module):
    """docstring for ILLoss"""
    def __init__(self):
        super(ILLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.Softmax = nn.Softmax(dim=1)
    def forward(self, outputs, labels):
        res = - torch.gather(self.LogSoftmax(outputs), dim=1, index=labels.unsqueeze(1)).squeeze(1)
        weights = self.Softmax(outputs).max(dim = 1)[0].data
        # weights = LOF_score(outputs)
        loss = (res * weights).mean()
        return loss

class Net(nn.Module):
    # define nn
    def __init__(self, X_dim, y_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_dim, 128)
        self.fc2 = nn.Linear(128, y_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.softmax(X)
        return X

def load_criterion(alg):
    if alg == 'CC':
        criterion = CCLoss(alpha = args.alpha)
    elif alg == 'NL':
        criterion = NLLoss()
    elif alg == 'SP':
        criterion =  SPLoss()
    elif alg == 'BT':
        criterion = BTLoss()
    elif alg == 'SCE':
        criterion = SCELoss()
    elif alg == 'MAE':
        criterion = MAELoss()
    elif alg == 'IL':
        criterion = ILLoss()
    elif alg == 'baseline':
        criterion =  nn.CrossEntropyLoss()
    else:
        criterion =  nn.CrossEntropyLoss()
    return criterion

def get_batch_size(num_ID, num_OOD):
    num_batch = int(num_ID / 128) + 1
    batch_size_ID, batch_size_OOD = int(num_ID / num_batch), int(num_OOD / num_batch)

    if batch_size_OOD == 0:
        batch_size_OOD = 1
    return batch_size_ID, batch_size_OOD

def generateOODLabel(inputs_OOD, y_dim):
    num_OOD = inputs_OOD.size(0)
    targets_OOD = torch.LongTensor(num_OOD).fill_(0)
    for i in range(num_OOD):
        targets_OOD[i] = int(abs(inputs_OOD[i].sum().item())) % y_dim
    return targets_OOD

def test_results(net, IDloader, OODloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(IDloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(inputs)
            softmax_vals, predicted = torch.max(F.softmax(out.data, dim=1), dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        test_acc = 100.0 * correct / total

    # soft_ID = np.array([])
    # soft_OOD = np.array([])
    # with torch.no_grad():
    #     for idx, (data_ID, data_OOD) in enumerate(zip(IDloader, OODloader)):
    #         inputs_ID, targets_ID = data_ID
    #         inputs_OOD, targets_OOD = data_OOD
    #         num_ID = inputs_ID.size(0)
    #         num_OOD = inputs_OOD.size(0)
    #         inputs = torch.cat((inputs_ID, inputs_OOD), dim = 0).cuda()
    #         outputs = net(inputs)
    #         softmax_vals, predicted = torch.max(F.softmax(outputs.data, dim=1), dim=1)
    #         softmax_vals = softmax_vals.cpu().numpy()
    #         soft_ID = np.append(soft_ID, softmax_vals[0:num_ID])
    #         soft_OOD = np.append(soft_OOD, softmax_vals[num_ID:num_ID + num_OOD])
    #     detection_results = OOD_Metric.detect_OOD(soft_ID, soft_OOD, precision=args.precision)
    detection_results = [0,0,0,0,0]
    ece, ys = OOD_Metric.Calibrated(IDloader, net)

    return test_acc, detection_results, ece, ys

def write_report(data, alg, seed, epoch, test_acc_OOD, detection_results, ece, ys):
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    tablename = ('results/' + alg + '.csv')

    if not os.path.exists(tablename):
        with open(tablename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['algorithm', 'dataset', 'seed', 'epoch', 'alpha',
                'ACC','ece', 'ys',
                'auroc','auprIn','auprOut','tpr95','detection'])

    with open(tablename, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([alg, data, seed, epoch, args.alpha,
            test_acc_OOD, ece, ys,
            detection_results[0],detection_results[1],detection_results[2],detection_results[3],detection_results[4]])

def main():
    # datasets = ["iris", "abalone", "wine", "covid19","gisette","skyserver",
    # "grid","bank","SHABD","IBM","gene","arrhythmia","speech","stellar"]
    # algs = ["baseline","MAE", "BT", "SP", "IL", "SCE", "NL", "CC"]

    torch.manual_seed(args.seed)

    X, y, X_dim, y_dim, OOD_index = Ndata.load_data(args.data)
    ID_X, ID_y, test_OOD_X, test_OOD_y = Ndata.ID_OOD_split(X, y, ood_label = OOD_index)
    train_ID_X, test_ID_X, train_ID_y, test_ID_y = train_test_split(ID_X, ID_y, test_size=args.rho)

    if args.normalization:
        train_ID_X, test_ID_X, test_OOD_X = Ndata.feature_normalize(train_ID_X, test_ID_X, test_OOD_X)
    Ndata.write_data(train_ID_X, train_ID_y, args.data, OOD_index, "train_ID")
    Ndata.write_data(test_ID_X, test_ID_y, args.data, OOD_index,"test_ID")
    Ndata.write_data(test_OOD_X, test_OOD_y, args.data, OOD_index,"test_OOD")

    batch_size_ID, batch_size_OOD = get_batch_size(len(train_ID_X), len(test_OOD_X))
    trainloader, testloader, OODloader, X_dim, y_dim = Ndata.get_loader_IOD(args.data, -1, 
        batch_size_ID = batch_size_ID, batch_size_OOD = batch_size_OOD)
            
    net = Net(X_dim, y_dim)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())
    criterion = load_criterion(args.alg)

    train_loss = 0.0
    correct = 0.0
    total = 0
    for epoch in range(args.epoch):
        for idx, (data_ID, data_OOD) in enumerate(zip(trainloader, OODloader)):
            inputs_ID, targets_ID = data_ID
            inputs_OOD, targets_OOD = data_OOD
            inputs_OOD, targets_OOD = data_OOD
            targets_OOD = generateOODLabel(inputs_OOD, y_dim)
            inputs = torch.cat((inputs_ID, inputs_OOD), dim = 0).cuda()
            targets = torch.cat((targets_ID, targets_OOD), dim = 0).cuda()

            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, targets)

            train_loss += loss.item()
            softmax_vals, predicted = torch.max(F.softmax(out.data, dim=1), dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            loss.backward()
            optimizer.step()
    train_loss = train_loss / (idx + 1)
    train_acc = 100.0 * correct / total
    test_acc, detection_results, ece, ys = test_results(net, testloader, OODloader)
    # print(args.data, args.alg, train_loss, train_acc, test_acc, detection_results, ece, ys)
    write_report(args.data, args.alg, args.seed, epoch, test_acc, detection_results, ece, ys)

if __name__ == '__main__':
    main()