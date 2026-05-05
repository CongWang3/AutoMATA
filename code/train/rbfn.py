import shutup  

shutup.please()

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import torch.optim as optim

import time

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score

from utils.FocalLoss import FocalLoss

from sklearn.model_selection import  StratifiedKFold

import pandas as pd

import os

import torch

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,classification_report,multilabel_confusion_matrix,hamming_loss

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.metrics import precision_recall_fscore_support

from utils.earlystopping import EarlyStopping  

from utils.regularization import apply_regularization_loss, apply_max_norm_constraint, create_optimizer_with_reg

import pickle

import argparse

import pickle

from DataProcess import load_data,process

def P_centers(X_train,centers_dim):
    kmeans = KMeans(n_clusters=centers_dim)
    kmeans.fit(X_train)  
    centers = kmeans.cluster_centers_  
    centers=torch.tensor(centers,dtype=torch.float32)
    distances = torch.cdist(X_train,centers)
    sigma = torch.std(distances, axis=0)  
    return centers,sigma

def train(dataloader, model):
    model.train()
    num_batches = len(dataloader)
    train_loss= 0 
    true_label_list, pred_label_list= [], []
    for data in dataloader:
        X_data, Y_data = data[0].to(device), data[1].to(device)  
        # print('X_data.shape =', X_data.shape)  # torch.Size([34, 4])
        # print('Y_data.shape =', Y_data.shape)  # torch.Size([34])
        output = model(X_data)  
        # print('output.shape =', output.shape)  # torch.Size([34, 2])
        loss = model.criterion(output, Y_data)  
        loss = apply_regularization_loss(loss, model, model.r_method, model.r_weight)
        train_loss += loss.item()  
        true_label_list.append(Y_data.cpu().detach().numpy())
        pred_label_list.append(output.argmax(dim=1).cpu().detach().numpy())  # softmax
        # pred_label_list.append(output.cpu().detach().numpy().round())  # sigmoid
        model.optimier.zero_grad()  
        loss.backward()  
        model.optimier.step()  
        if model.r_method == "maxnorm":
            apply_max_norm_constraint(model, model.r_weight)
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    # print('y_pred =', y_pred)
    train_loss /= num_batches  
    train_acc = accuracy_score(y_true,y_pred)
    train_precision, train_recall, train_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    return train_loss, train_acc, train_precision, train_recall, train_f1

def validate(dataloader, model):
    size = len(dataloader.dataset)  
    num_batches = len(dataloader)  
    # print('num_batches =', num_batches)
    model.eval()
    val_loss = 0 
    true_label_list, pred_label_list= [], []
    with torch.no_grad(): 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  
            pred = model(X)
            val_loss += model.criterion(pred, y).item() 
            true_label_list.append(y.cpu().detach().numpy())
            pred_label_list.append(pred.argmax(dim=1).cpu().detach().numpy())
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    val_loss /= num_batches  
    val_acc = accuracy_score(y_true,y_pred)
    val_precision, val_recall, val_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    return val_loss, val_acc, val_precision, val_recall, val_f1

def test(dataloader, model):
    size = len(dataloader.dataset)  
    num_batches = len(dataloader)  
    # print('num_batches =', num_batches)
    model.eval()
    true_label_list, pred_label_list= [], []
    with torch.no_grad(): 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  
            pred = model(X)
            true_label_list.append(y.cpu().detach().numpy())
            pred_label_list.append(pred.argmax(dim=1).cpu().detach().numpy())
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    acc = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    return acc, precision, recall, f1

class RBFN(nn.Module):
    def __init__(self,centers_dim,out_dim,centers,sigma,dropout_rate=0.0, r_method=None, r_weight=0.0):
        super(RBFN,self).__init__()
        self.flatten = nn.Flatten()
        self.centers_dim=centers_dim
        self.out_dim=out_dim
        self.r_method = r_method
        self.r_weight = r_weight
        self.centers = nn.Parameter(centers)
        self.sigma = nn.Parameter(sigma)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.centers_dim, self.out_dim)
        self.learning_rate = lr
        self.loss_function = loss_function  
        if loss_function == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "focalloss":
            self.criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-class', num_classes=out_dim)
        elif loss_function == "nllloss":
            self.criterion = nn.NLLLoss()
        self.optimier = create_optimizer_with_reg(self, optimizer_function, lr, r_method, r_weight)
    
    def forward(self,X):
        x= self.flatten(X)
        distance = torch.cdist(x, self.centers)
        gauss = torch.exp(-distance ** 2 / (2 * self.sigma ** 2))
        gauss = self.dropout(gauss)
        y=self.linear(gauss)
        if self.loss_function == "nllloss":
            return nn.functional.log_softmax(y, dim=1)
        return y

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--kfold", default=0, type=int)  
    parser.add_argument("--ratio", default="0", type=str)  
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--es", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--loss_function", default="crossentropy", type=str)  
    parser.add_argument("--optimizer_function", default="adam", type=str)  # adam, rmsprop, sgd
    parser.add_argument("--output_size", default=2, type=int)  
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')  
    parser.add_argument('--r_method', type=str, default=None, help='Regularization method: l1, l2, maxnorm, sparsity, or none')
    parser.add_argument('--r_weight', type=float, default=0.0, help='Regularization weight/strength')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--feature_method', type=str, default=None, help='Feature selection method: PCC, SPEARMAN, CHI2, RF, etc.')
    args = parser.parse_args()
    kfold = args.kfold
    ratio = args.ratio
    epochs = args.epochs
    es = args.es
    lr = args.lr
    batch_size = args.bs
    loss_function = args.loss_function
    optimizer_function = args.optimizer_function
    out_dim = args.output_size  
    type = args.type
    r_method = args.r_method if args.r_method and args.r_method != "none" else None
    r_weight = args.r_weight
    dropout_rate = args.dropout_rate
    feature_method = args.feature_method if args.feature_method and args.feature_method.strip() else None
    print('model = RBFN')
    print('kfold =', kfold)
    print('ratio =', ratio)
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', out_dim)
    random_seed = args.random_seed  
    print('random seed =', random_seed)  
    print('regularization method =', r_method if r_method else 'none')
    print('regularization weight =', r_weight)
    print('dropout rate =', dropout_rate)
    print('feature selection method =', feature_method if feature_method else 'none')
    set_random_seed(random_seed)  
    centers_dim = 10 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_path = "./result/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savename = result_path + 'model.pth'

    # savename = 'D:\\wamp\www\\multi_omics_own\\download_data\\Jobs\\'+jobID+'\\result\\model.pt'
    early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)
    if (ratio != "0"):
        process(ratio=ratio)
    '''Train the model'''
    X_train, Y_train, feature_indices = load_data("train", feature_method=feature_method)
    actual_num_classes = len(torch.unique(Y_train))
    if actual_num_classes != out_dim:
        print(f"Warning: The number of classes set by the user ({out_dim}) does not match the actual number of classes in the data ({actual_num_classes})")
        print(f"Automatically use the actual number of classes: {actual_num_classes}")
        out_dim = actual_num_classes
    # n_samples should be >= n_clusters
    if (centers_dim > X_train.shape[0]):
        centers_dim = X_train.shape[0]
    input_dim = X_train.shape[1]
    if kfold:
        kfscore = []
        skf = StratifiedKFold(n_splits=kfold)  
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
            print("--------The {} fold is training---------".format(i+1))
            trainset, valset = torch.FloatTensor(np.array(X_train)[train_idx]), torch.FloatTensor(np.array(X_train)[val_idx])
            traintag, valtag = torch.LongTensor(np.array(Y_train)[train_idx]), torch.LongTensor(np.array(Y_train)[val_idx])
            train_dataset = torch.utils.data.TensorDataset(trainset, traintag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset =  torch.utils.data.TensorDataset(valset, valtag)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            centers,sigma=P_centers(X_train,centers_dim)
            # define model
            model= RBFN(centers_dim,out_dim,centers,sigma,dropout_rate, r_method, r_weight).to(device)
            val_acc_s = []  
            val_loss_s = []  
            train_acc_s = []  
            train_loss_s = []  
            
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=model)
                val_loss, val_acc, val_precision, val_recall, val_f1  = validate(dataloader=val_loader, model=model)
                print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
                print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
                train_acc_s.append(train_acc)
                train_loss_s.append(train_loss)
                val_acc_s.append(val_acc)
                val_loss_s.append(val_loss)
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print('early stopping')
                    epochs = t+1  
                    break
            print("--------The {} fold validation result---------".format(i+1))
            val_acc, val_precision, val_recall, val_f1 = test(dataloader=val_loader, model=model)
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            kfscore.append(test(dataloader=val_loader, model=model))
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[0], 4), round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4)))
    else:
        train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val, _ = load_data("validate", feature_indices=feature_indices)
        val_dataset =  torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        centers,sigma=P_centers(X_train,centers_dim)
        model= RBFN(centers_dim,out_dim,centers,sigma,dropout_rate, r_method, r_weight).to(device)
        val_acc_s = []  
        val_loss_s = []  
        train_acc_s = []  
        train_loss_s = []  

        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=model)
            val_loss, val_acc, val_precision, val_recall, val_f1  = validate(dataloader=val_loader, model=model)
            print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
            print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            train_acc_s.append(train_acc)
            train_loss_s.append(train_loss)
            val_acc_s.append(val_acc)
            val_loss_s.append(val_loss)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('early stopping')
                epochs = t+1  
                break

    '''Draw the training-validation curve'''
    plt.plot(list(range(1, epochs+1)), train_loss_s, label = 'training loss')  
    plt.plot(list(range(1, epochs+1)), val_loss_s, label = 'validation loss')  
    plt.plot(list(range(1, epochs+1)), train_acc_s, label = 'training accuracy')  
    plt.plot(list(range(1, epochs+1)), val_acc_s, label = 'validation accuracy')  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss—Accuracy")
    plt.title('acc-loss curve')
    plt.legend(loc='upper left')
    # plt.savefig("D:\\wamp\\www\\multi_omics_own\\download_data\\Jobs\\"+jobID+"\\result\\figure"+".png", format='png')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()
    print("Done!")  

    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(), 
        'loss_function': loss_function,
        'optimizer_function': optimizer_function,
        'centers_dim': centers_dim,
        'input_dim': input_dim,
        'out_dim': out_dim,
        'dropout_rate': dropout_rate,
        'r_method': r_method,
        'r_weight': r_weight,
        'learning_rate': lr,
        'centers': centers,
        'sigma': sigma,
        'feature_indices': feature_indices
    }, savename)

    '''Test the model'''
    X_test, Y_test, _ = load_data("test", feature_indices=feature_indices)
    test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    acc, precision, recall, f1 = test(dataloader=test_loader, model=model)
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))
    # with open("D:\\wamp\\www\\multi_omics_own\\download_data\\Jobs\\"+jobID+"\\result\\"+"test_result.txt", mode="w") as f:
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")
