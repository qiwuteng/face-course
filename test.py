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
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = True

def plot_confusion_matrix(cm,
                          savename,
                          target_names,
                          title='Confusion matrix',
                          cmap='Blues',
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    error = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(9, 7))
#    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "red")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('True label',size=15)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; error={:0.4f}'.format(accuracy, error),size=15)
    plt.savefig(savename, format='png', bbox_inches = 'tight')
    plt.show()


def test(model, test_iter, device, loss, attributes):
    model.eval()
    test_loss = 0
    Average_Precisions = []
    labels = {}
    Acc = []
    probabilities = {}
    predicts = {}
    correct = {}
    for attr in attributes:
        labels[attr] = list()
        probabilities[attr] = list()
        predicts[attr] = list()
        correct[attr] = 0
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (X, Y) in enumerate(test_iter):
            test_iter_loss = 0
            X = X.to(device)
            Y = [y.to(device) for y in Y]
            Y_hat = list(model(X))
            for j in range(len(Y)):
                test_iter_loss += loss(Y_hat[j], Y[j])
                Y_hat[j] = softmax(Y_hat[j])
                # print(torch.max(Y[j],1)[1].cpu().numpy().tolist())
                labels[attributes[j]] += Y[j].cpu().numpy().tolist()
                probabilities[attributes[j]] += Y_hat[j][:, 1].cpu().numpy().tolist()
                predicts[attributes[j]] += torch.max(Y_hat[j], 1)[1].cpu().numpy().tolist()
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

    return labels, predicts


def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    model = cfg.model
    batch_size = cfg.batch_szie
    num_workers = cfg.num_workers
    test_json = cfg.test_json_path
    attributes = cfg.attributes
    checkpoint_path = cfg.checkpoint_path
    cm_path = cfg.cm_path
    device = torch.device("cuda")
    print('test on', device)

    test_iter = load_data(test_json, attributes , batch_size, num_workers, mode='test')
    
    pt_path = os.path.join(checkpoint_path + model + '.pt')
    net = Model(model, True)
    net.load_state_dict(torch.load(pt_path))
    net.to(device)

    loss = nn.CrossEntropyLoss()
  
    labels, predicts = test(net, test_iter, device, loss, cfg.attributes)

    if not os.path.exists(cm_path):
        os.makedirs(cm_path)

    for i in range(len(attributes)):
        attr = attributes[i]
        save_path = cm_path + '/' + model +'_' + attr + '.png'
        title = model + ' confusion matrix'
        cm = confusion_matrix(labels[attr], predicts[attr])
        classes = ['0', '1']
        plot_confusion_matrix(cm, save_path, classes, title)

if __name__ == "__main__":
    main()