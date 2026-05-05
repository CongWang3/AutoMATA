import shutup  
shutup.please()
import math
import warnings
import numpy as np
import torch.optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
from utils.FocalLoss import FocalLoss
from sklearn.model_selection import  StratifiedKFold
import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,classification_report,multilabel_confusion_matrix,hamming_loss
from utils.plot_utils import plotfig
from utils.earlystopping import EarlyStopping  
from utils.regularization import apply_regularization_loss, apply_max_norm_constraint, create_optimizer_with_reg

warnings.simplefilter(action='ignore', category=RuntimeWarning)

torch.manual_seed(2022)

import argparse

import pickle

from DataProcess import load_data,process

def train(dataloader, model):
    model.train()
    num_batches = len(dataloader)
    train_loss= 0 
    true_label_list, pred_label_list= [], []
    for data in dataloader:
        X_data, Y_data = data[0].to(device), data[1].to(device)  
        output = model(X_data)  
        loss = model.criterion(output, Y_data)  
        loss = apply_regularization_loss(loss, model, model.r_method, model.r_weight)
        train_loss += loss.item()  
        true_label_list.append(Y_data.cpu().detach().numpy())
        pred_label_list.append(output.argmax(dim=1).cpu().detach().numpy())
        model.optimier.zero_grad()  
        loss.backward()  
        model.optimier.step()  
        if model.r_method == "maxnorm":
            apply_max_norm_constraint(model, model.r_weight)
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
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

class Cnn(nn.Module):
    def __init__(self, conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function, r_method=None, r_weight=0.0):
        super(Cnn, self).__init__()
        self.output_size = output_size  
        self.r_method = r_method
        self.r_weight = r_weight
        self.model1 = nn.Sequential(
            nn.Conv1d(1, conv_size_1, 2, stride=2, padding=6),  # nn.Conv1d(in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([34, 16, 31])
        )
        self.model2 = nn.Sequential(
            nn.Conv1d(conv_size_1, conv_size_2, 2, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(4),  # torch.Size([34, 32, 7])
            nn.Flatten(),  
            nn.Dropout(dropout_rate)  
        )
        self.learning_rate = lr
        self.fc = nn.LazyLinear(self.output_size)  
        self.loss_function = loss_function  
        if loss_function == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "focalloss":
            self.criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-class', num_classes=output_size)
        elif loss_function == "nllloss":
            self.criterion = nn.NLLLoss()
        self.optimier = create_optimizer_with_reg(self, optimizer_function, lr, r_method, r_weight)
    
    def forward(self, input):
        # print(cnn)
        # print('input_1 shape =', input.shape)  # torch.Size([34, 4])
        input = input.reshape(-1,1,input.shape[1])   
        # print('input_2 shape =', input.shape)
        x = self.model1(input)
        # print('x_1 model1 shape =', x.shape)  # torch.Size([34, 224])
        x = self.model2(x)
        # print('x_2 model1 shape =', x.shape)  # torch.Size([34, 224])
        # fc = nn.Linear(in_features=x.shape[1], out_features=self.output_size, bias=True).to(device)
        # x = fc(x)
        # if (self.output_size == 2):
        #     act = nn.Softmax(dim=1).to(device)
        #     x = act(x)
        # return x
        logits = self.fc(x)
        if self.loss_function == "nllloss":
            return nn.functional.log_softmax(logits, dim=1)
        return logits

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
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')  
    parser.add_argument('--r_method', type=str, default=None, help='Regularization method: l1, l2, maxnorm, sparsity, or none')
    parser.add_argument('--r_weight', type=float, default=0.0, help='Regularization weight/strength')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--feature_method', type=str, default=None, help='Feature selection method: PCC, SPEARMAN, CHI2, RF.')
    args = parser.parse_args()
    kfold = args.kfold
    ratio = args.ratio
    epochs = args.epochs
    es = args.es
    lr = args.lr
    batch_size = args.bs
    loss_function = args.loss_function
    optimizer_function = args.optimizer_function
    output_size = args.output_size  
    type = args.type
    random_seed = args.random_seed  
    r_method = args.r_method if args.r_method and args.r_method != "none" else None
    r_weight = args.r_weight
    dropout_rate = args.dropout_rate
    feature_method = args.feature_method if args.feature_method and args.feature_method.strip() else None
    print('model = CNN')
    print('kfold =', kfold)
    print('ratio =', ratio)
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', output_size)
    print('random seed =', random_seed)  
    print('regularization method =', r_method if r_method else 'none')
    print('regularization weight =', r_weight)
    print('dropout rate =', dropout_rate)
    print('feature selection method =', feature_method if feature_method else 'none')
    
    conv_size_1 = 64
    conv_size_2 = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(random_seed)  

    result_path = "./result/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savename = result_path + 'model.pth'
    early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)
    if (ratio != "0"):
        process(ratio=ratio)
    X_train, Y_train, feature_indices = load_data("train", feature_method=feature_method)
    
    input_size = X_train.shape[1]
    actual_num_classes = len(torch.unique(Y_train))
    if actual_num_classes != output_size:
        print(f"Warning: The number of classes set by the user ({output_size}) does not match the actual number of classes in the data ({actual_num_classes})")
        print(f"Automatically use the actual number of classes: {actual_num_classes}")
        output_size = actual_num_classes

    '''training model'''
    if (kfold):
        kfscore = []
       
        skf = StratifiedKFold(n_splits=kfold)  
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
            early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)  
            print("--------The {} fold is training---------".format(i+1))
            trainset, valset = torch.FloatTensor(np.array(X_train)[train_idx]), torch.FloatTensor(np.array(X_train)[val_idx])
            traintag, valtag = torch.LongTensor(np.array(Y_train)[train_idx]), torch.LongTensor(np.array(Y_train)[val_idx])
            # print(trainset)
            # print(traintag)
            train_dataset = torch.utils.data.TensorDataset(trainset, traintag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset =  torch.utils.data.TensorDataset(valset, valtag)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            # define model
            cnn = Cnn(conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function, r_method, r_weight).to(device)
            val_acc_s = []  
            val_loss_s = []  
            train_acc_s = []  
            train_loss_s = []  
            
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=cnn)
                val_loss, val_acc, val_precision, val_recall, val_f1 = validate(dataloader=val_loader, model=cnn)
                print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
                print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
                train_acc_s.append(train_acc)
                train_loss_s.append(train_loss)
                val_acc_s.append(val_acc)
                val_loss_s.append(val_loss)
                early_stopping(val_loss, cnn)
                if early_stopping.early_stop:
                    print('early stopping')
                    epochs = t+1  
                    break
            print("--------The {} fold validation result---------".format(i+1))
            val_acc, val_precision, val_recall, val_f1 = test(dataloader=val_loader, model=cnn)
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            kfscore.append(test(dataloader=val_loader, model=cnn))
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[0], 4), round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4)))
    
    else:
        cnn = Cnn(conv_size_1, conv_size_2, output_size, dropout_rate, lr, loss_function, optimizer_function, r_method, r_weight).to(device)
        train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val, _ = load_data("validate", feature_indices=feature_indices)
        val_dataset =  torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        val_acc_s = []  
        val_loss_s = []  
        train_acc_s = []  
        train_loss_s = []  
        
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss, train_acc, train_precision, train_recall, train_f1 = train(dataloader=train_loader, model=cnn)
            val_loss, val_acc, val_precision, val_recall, val_f1  = validate(dataloader=val_loader, model=cnn)
            print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
            print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            train_acc_s.append(train_acc)
            train_loss_s.append(train_loss)
            val_acc_s.append(val_acc)
            val_loss_s.append(val_loss)
            early_stopping(val_loss, cnn)
            if early_stopping.early_stop:
                print('early stopping')
                epochs = t+1  
                break
    
    
    torch.save({
        'epochs': epochs,
        'model_state_dict': cnn.state_dict(), 
        'loss_function': loss_function,
        'optimizer_function': optimizer_function,
        'output_size': output_size,
        'conv_size_1': conv_size_1,
        'conv_size_2': conv_size_2,
        'dropout_rate': dropout_rate,
        'learning_rate': lr,
        'input_size': input_size,
        'r_method': r_method,
        'r_weight': r_weight,
        'feature_indices': feature_indices,
    }, savename)
    
    '''plot training-validation curve'''
    plt.plot(list(range(1, epochs+1)), train_loss_s, label = 'training loss')  
    plt.plot(list(range(1, epochs+1)), val_loss_s, label = 'validation loss')  
    plt.plot(list(range(1, epochs+1)), train_acc_s, label = 'training accuracy')  
    plt.plot(list(range(1, epochs+1)), val_acc_s, label = 'validation accuracy')  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss—Accuracy")
    plt.title('acc-loss curve')
    plt.legend(loc='upper left')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()
    print("Done!")  
    
    '''test model'''
    X_test, Y_test, _ = load_data("test", feature_indices=feature_indices)
    test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    acc, precision, recall, f1 = test(dataloader=test_loader, model=cnn)
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))
    
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")