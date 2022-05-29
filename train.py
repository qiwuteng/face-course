from genericpath import exists
from FS2K import load_data
from model import Model
import random
import numpy as np
import config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from sklearn import metrics

from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = True

def train(model, train_iter, device, optimizer, loss, epoch):
    model.train()
    train_loss = 0
    for i, (X, Y) in enumerate(train_iter):
        train_iter_loss = 0
        optimizer.zero_grad()
        X = X.to(device)
        Y = [y.to(device) for y in Y]
        Y_hat = list(model(X))
        for j in range(len(Y)):
            train_iter_loss += loss(Y_hat[j], Y[j])
        train_loss += train_iter_loss
        train_iter_loss.backward()
        optimizer.step()
    print("Train Epoch: {}, Total Loss: {}".format(epoch, train_loss))


def test(model, test_iter, device, loss, attributes):
    model.eval()
    test_loss = 0
    Average_Precisions = []
    Acc = []
    labels = {}
    probabilities = {}
    correct = {}
    for attr in attributes:
        labels[attr] = list()
        probabilities[attr] = list()
        correct[attr] = 0
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (X, Y) in enumerate(test_iter):
            test_iter_loss = 0
            X = X.to(device)
            Y = [y.to(device) for y in Y]
            Y_hat = list(model(X))
            for j in range(len(Y)):
                Y_hat[j] = softmax(Y_hat[j])
                test_iter_loss += loss(Y_hat[j], Y[j])
                labels[attributes[j]] += Y[j].cpu().numpy().tolist()
                probabilities[attributes[j]] += Y_hat[j][:, 1].cpu().numpy().tolist()
                Y_hat_idx = torch.max(Y_hat[j], 1)[1]
                correct[attributes[j]] += torch.eq(Y_hat_idx, Y[j]).sum()
            test_loss += test_iter_loss
    
    for attr in attributes:
        Average_Precisions.append(metrics.average_precision_score(labels[attr], probabilities[attr]))
        Acc.append(correct[attr]/len(test_iter.dataset))

    print("Test total Loss: {}".format(test_loss))
    for i in range(len(Average_Precisions)):
        print("{} AP: {}, Acc: {}".format(attributes[i], Average_Precisions[i], Acc[i]))
    mAP = sum(Average_Precisions)/len(Average_Precisions)
    mAcc = sum(Acc)/len(Acc)
    print("mAP: {}, mAcc: {}".format(mAP, mAcc))

    return mAcc

def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    model = cfg.model
    batch_size = cfg.batch_szie
    num_workers = cfg.num_workers
    num_epochs = cfg.epochs
    lr = cfg.lr
    train_json = cfg.train_json_path
    test_json = cfg.test_json_path
    attributes = cfg.attributes
    checkpoint_path = cfg.checkpoint_path
    device = torch.device("cuda")
    print('training on', device)

    train_iter = load_data(train_json, attributes , batch_size, num_workers, mode='train')
    test_iter = load_data(test_json, attributes , batch_size, num_workers, mode='test')
    
    net = Model(model, True)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    pt_path = os.path.join(checkpoint_path + model + '.pt')
    mAcc = 0
    for epoch in range(num_epochs):
        train(net, train_iter, device, optimizer, loss, epoch)
        # print("Train Epoch: {}, Total Loss: {}".format(epoch, train_loss))
        if (epoch+1) % 10 == 0:
            tmp = test(net, test_iter, device, loss, cfg.attributes)
            if tmp > mAcc:
                mAcc= tmp
                torch.save(net.state_dict(), pt_path)

if __name__ == "__main__":
    main()

