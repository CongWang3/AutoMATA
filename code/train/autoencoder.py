import shutup  

shutup.please()

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import os

import pickle

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import  StratifiedKFold

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,classification_report,multilabel_confusion_matrix,hamming_loss

from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder

from utils.earlystopping import EarlyStopping

from utils.FocalLoss import FocalLoss

from utils.regularization import apply_regularization_loss, apply_max_norm_constraint, create_optimizer_with_reg

import argparse

import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

torch.manual_seed(2022)

from DataProcess import load_data,process  

class Autoencoder(nn.Module):  # 64, 32 ,16
    def __init__(self, input_dim, hidden_size_1=64, hidden_size_2=32, hidden_size_3=16, lr=0.001, dropout_rate=0.5):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(True),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size_3, hidden_size_2),
            nn.ReLU(True),
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(True),
            nn.Linear(hidden_size_1, input_dim),
        )
        self.criterion = nn.MSELoss()  # Autoencoder usually uses MSELoss to optimise the reconstruction error.
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class Classifier(nn.Module):  # 16, 8, 2
    def __init__(self, hidden_size_3=16, cls_hidden_size=8, output_size=2, lr=0.001, loss_function="crossentropy", optimizer_function="adam", r_method=None, r_weight=0.0):
        self.output_size = output_size  # Record the output dimension of the flatten layer
        super(Classifier, self).__init__()
        self.r_method = r_method
        self.r_weight = r_weight
        self.fc = nn.Sequential(
            nn.Linear(hidden_size_3, cls_hidden_size),
            nn.ReLU(),
            nn.Linear(cls_hidden_size, output_size),
        )
        self.learning_rate = lr
        self.loss_function = loss_function  
        if loss_function == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "focalloss":
            self.criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-class', num_classes=output_size)
        elif loss_function == "nllloss":
            self.criterion = nn.NLLLoss()
        # Create optimizer with regularization support
        self.optimizer = create_optimizer_with_reg(self, optimizer_function, lr, r_method, r_weight)
    def forward(self, x):
        logits = self.fc(x)
        if self.loss_function == "nllloss":
            return nn.functional.log_softmax(logits, dim=1)
        return logits
# train Autoencoder

def train_autoencoder(model, dataloader):
    train_loss = 0
    for data in dataloader:
        inputs, _ = data[0].to(device), data[1].to(device)
        model.optimizer.zero_grad()
        decoded, _ = model(inputs)
        loss = model.criterion(decoded, inputs)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader)  # Loss of one epoch
    return train_loss
# validate Autoencoder

def val_autoencoder(model, val_dataloader):
    val_loss = 0
    with torch.no_grad(): # Context manager that disables gradient computation in the code blocks it wraps to reduce memory usage and speed up computation.
        for data in val_dataloader:
            inputs, _ = data[0].to(device), data[1].to(device)
            decoded, _  = model(inputs)
            val_loss += model.criterion(decoded, inputs).item() # Calculate and accumulate losses
    val_loss /= len(val_dataloader)  # Loss of one epoch
    return val_loss
# train classifier

def train_classifier(classifier, dataloader):
    true_label_list, pred_label_list= [], []
    train_loss = 0
    classifier.train()
    for data in dataloader:
        features, labels = data[0].to(device), data[1].to(device)
        classifier.optimizer.zero_grad()
        outputs = classifier(features)
        loss = classifier.criterion(outputs, labels)
        # Apply regularization penalties (except max-norm which is applied after optimizer step)
        loss = apply_regularization_loss(loss, classifier, classifier.r_method, classifier.r_weight)
        loss.backward()
        classifier.optimizer.step()
        # Apply max-norm constraint if specified
        if classifier.r_method == "maxnorm":
            apply_max_norm_constraint(classifier, classifier.r_weight)
        train_loss += loss.item()
        true_label_list.append(labels.cpu().detach().numpy())
        pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy())
    train_loss /= len(dataloader)
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    train_acc = accuracy_score(y_true,y_pred)
    train_precision, train_recall, train_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    return train_loss, train_acc, train_precision, train_recall, train_f1
# validate classifier

def val_classifier(classifier, dataloader):
    true_label_list, pred_label_list= [], []
    classifier.eval()
    val_loss = 0 
    with torch.no_grad():
        for data in dataloader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(features)
            loss = classifier.criterion(outputs, labels)
            val_loss += loss.item()
            true_label_list.append(labels.cpu().detach().numpy())
            pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy()) 
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    val_loss /= len(dataloader)
    val_acc = accuracy_score(y_true,y_pred)
    val_precision, val_recall, val_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    return val_loss, val_acc, val_precision, val_recall, val_f1
def extract_features(autoencoder, dataloader):
    encoder = autoencoder.encoder
    features = []
    labels = []
    for data in dataloader:
        inputs, label = data
        inputs, label  = inputs.to(device), label.to(device)
        with torch.no_grad():
            encoded_features = encoder(inputs)
        features.append(encoded_features)
        labels.append(label)
    # concatenate features and labels
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    # construct dataloader
    dataset =  torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader
# save models

def save_model(autoencoder, classifier, autoencoder_path, classifier_path):
    torch.save(autoencoder.state_dict(), autoencoder_path)
    torch.save(classifier.state_dict(), classifier_path)
    print("model save successfully!")
# load models

def load_model(autoencoder, classifier, autoencoder_path, classifier_path):
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    classifier.load_state_dict(torch.load(classifier_path))
    print("model load successfully!")

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# test model
def test(dataloader, autoencoder, classifier):
    loader = extract_features(autoencoder, dataloader) 
    true_label_list, pred_label_list= [], []
    classifier.eval()
    with torch.no_grad():
        for data in loader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(features)
            true_label_list.append(labels.cpu().detach().numpy())
            pred_label_list.append(outputs.argmax(dim=1).cpu().detach().numpy())
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    acc = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    return acc, precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--kfold", default=0, type=int)  
    parser.add_argument("--ratio", default="0", type=str)  
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--es", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--loss_function", default="crossentropy", type=str)  
    parser.add_argument("--optimizer_function", default="adam", type=str)  # adam, rmsprop, sgd
    parser.add_argument("--output_size", default=4, type=int)  
    parser.add_argument("--type", default="single", type=str)  
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
    print('model = AutoEncoder')
    print('kfold =', kfold)
    print('ratio =', ratio) 
    print('epochs =', epochs)
    print('earlystopping =', es)
    print('learning rate =', lr)
    print('batch size =', batch_size)
    print('loss function =', loss_function)
    print('optimizer function =', optimizer_function)
    print('label number =', output_size)
    random_seed = args.random_seed  
    print('random seed =', random_seed)  
    set_random_seed(random_seed) 
    random_seed = args.random_seed 
    r_method = args.r_method if args.r_method and args.r_method != "none" else None
    r_weight = args.r_weight
    dropout_rate = args.dropout_rate
    feature_method = args.feature_method if args.feature_method and args.feature_method.strip() else None
    print('regularization method =', r_method if r_method else 'none')
    print('regularization weight =', r_weight)
    print('dropout rate =', dropout_rate)
    print('feature selection method =', feature_method if feature_method else 'none')
    
    hidden_size_1=64
    hidden_size_2=32
    hidden_size_3=16
    cls_hidden_size=8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_path = "./result/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savename = result_path+'model_autoencoder.pth'
    savename_cls = result_path+'model_cls.pth'
    early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)
    early_stopping_cls = EarlyStopping(es, verbose=True, savename=savename_cls, delta=0.0001)
    if (ratio != "0"):
        process(ratio=ratio)
    X_train, Y_train, feature_indices = load_data("train", feature_method=feature_method)
    actual_num_classes = len(torch.unique(Y_train))
    if actual_num_classes != output_size:
        print(f"Warning: The number of classes set by the user ({output_size}) does not match the actual number of classes in the data ({actual_num_classes})")
        print(f"Automatically use the actual number of classes: {actual_num_classes}")
        output_size = actual_num_classes
    input_dim = X_train.shape[1]
    
    if (kfold):
        kfscore = []
        skf = StratifiedKFold(n_splits=kfold)
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
            early_stopping = EarlyStopping(es, verbose=True, savename=savename, delta=0.0001)  
            early_stopping_cls = EarlyStopping(es, verbose=True, savename=savename_cls, delta=0.0001)
            print("--------The {} fold is training---------".format(i+1))
            trainset, valset = torch.FloatTensor(np.array(X_train)[train_idx]), torch.FloatTensor(np.array(X_train)[val_idx])
            traintag, valtag = torch.LongTensor(np.array(Y_train)[train_idx]), torch.LongTensor(np.array(Y_train)[val_idx])
            train_dataset = torch.utils.data.TensorDataset(trainset, traintag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset =  torch.utils.data.TensorDataset(valset, valtag)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            # define model
            autoencoder = Autoencoder(input_dim, hidden_size_1, hidden_size_2, hidden_size_3, lr, dropout_rate).to(device)
            classifier = Classifier(hidden_size_3, cls_hidden_size, output_size, lr, loss_function, optimizer_function, r_method, r_weight).to(device)  
            # begin training
            val_loss_s = [] 
            train_loss_s = []
            print("--------Training Autoencoder--------")
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                train_loss = train_autoencoder(autoencoder, train_loader)
                val_loss = val_autoencoder(autoencoder, val_loader)
                print("train loss = {}".format(train_loss))
                print("validation loss = {}".format(val_loss))
                train_loss_s.append(train_loss)
                val_loss_s.append(val_loss)
                early_stopping(val_loss, autoencoder)
                if early_stopping.early_stop:
                    print('early stopping')
                    epochs = t+1 
                    break
            print("--------Autoencoder Training Ends--------")
            val_acc_s_cls = []  
            val_loss_s_cls = []
            train_acc_s_cls = []
            train_loss_s_cls = [] 
            print("--------Training Classifier--------")
            train_loader = extract_features(autoencoder, train_loader)
            val_loader = extract_features(autoencoder, val_loader)
            for t in range(epochs):
                print("--------Begin the {} epoch training---------".format(t+1))
                train_loss, train_acc, train_precision, train_recall, train_f1 = train_classifier(classifier, train_loader)
                val_loss, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
                print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
                print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
                train_loss_s_cls.append(train_loss)
                val_loss_s_cls.append(val_loss)
                val_acc_s_cls.append(val_acc)
                train_acc_s_cls.append(train_acc)
                early_stopping_cls(val_loss, classifier)
                if early_stopping_cls.early_stop:
                    print('early stopping')
                    epochs = t+1 
                    break
            print("--------Classifier Training Ends--------")
            print("--------The {} fold validation result---------".format(i+1))
            _, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            kfscore.append(val_classifier(classifier, val_loader))
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4), round(kfscore[4], 4)))
    else:
        autoencoder = Autoencoder(input_dim, hidden_size_1, hidden_size_2, hidden_size_3, lr, dropout_rate).to(device)
        classifier = Classifier(hidden_size_3, cls_hidden_size, output_size, lr, loss_function, optimizer_function, r_method, r_weight).to(device)  
        train_dataset =  torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val, _ = load_data("validate", feature_indices=feature_indices)
        val_dataset =  torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        val_loss_s = [] 
        train_loss_s = [] 
        print("--------Training Autoencoder--------")
        
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss = train_autoencoder(autoencoder, train_loader)
            val_loss = val_autoencoder(autoencoder, val_loader)
            print("train loss = {}".format(train_loss))
            print("validation loss = {}".format(val_loss))
            train_loss_s.append(train_loss)
            val_loss_s.append(val_loss)
            early_stopping(val_loss, autoencoder)
            if early_stopping.early_stop:
                print('early stopping')
                epochs = t+1 
                break
        print("--------Autoencoder Training Ends--------")
        val_acc_s_cls = []
        val_loss_s_cls = [] 
        train_acc_s_cls = [] 
        train_loss_s_cls = [] 
        print("--------Training Classifier--------")
        train_loader = extract_features(autoencoder, train_loader)
        val_loader = extract_features(autoencoder, val_loader)
        
        for t in range(epochs):
            print("--------Begin the {} epoch training---------".format(t+1))
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_classifier(classifier, train_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1 = val_classifier(classifier, val_loader)
            print("train loss = {}, acc = {}, precision = {}, recall = {}, f1 = {} ".format(round(train_loss, 4), round(train_acc, 4), round(train_precision, 4), round(train_recall, 4), round(train_f1, 4)))
            print("validation loss = {}, acc = {}, precision = {}, recall = {}, f1 = {}".format(round(val_loss, 4), round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(val_f1, 4)))
            train_loss_s_cls.append(train_loss)
            val_loss_s_cls.append(val_loss)
            val_acc_s_cls.append(val_acc)
            train_acc_s_cls.append(train_acc)
            early_stopping_cls(val_loss, classifier)
            if early_stopping_cls.early_stop:
                print('early stopping')
                epochs = t+1
                break
        print("--------Classifier Training Ends--------")
    
    plt.plot(list(range(1, epochs+1)), train_loss_s_cls, label = 'training loss') 
    plt.plot(list(range(1, epochs+1)), val_loss_s_cls, label = 'validation loss')
    plt.plot(list(range(1, epochs+1)), train_acc_s_cls, label = 'training accuracy')
    plt.plot(list(range(1, epochs+1)), val_acc_s_cls, label = 'validation accuracy')
    plt.xlabel("Epoch") 
    plt.ylabel("Loss—Accuracy")
    plt.title('acc-loss curve')
    plt.legend(loc='upper left')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()
    print("Done!")
    
    torch.save({
        'epochs': epochs,
        'model_state_dict': autoencoder.state_dict(), 
        'input_dim': input_dim,
        'hidden_size_1': hidden_size_1,
        'hidden_size_2': hidden_size_2,
        'hidden_size_3': hidden_size_3,
        'learning_rate': lr,
        'dropout_rate': dropout_rate,
        'feature_indices': feature_indices
    }, savename)
    
    torch.save({
        'model_state_dict': classifier.state_dict(), 
        'hidden_size_3': hidden_size_3,
        'cls_hidden_size': cls_hidden_size,
        'output_size': output_size,
        'learning_rate': lr,
        'loss_function': loss_function,
        'optimizer_function': optimizer_function,
        'r_method': r_method,
        'r_weight': r_weight,
        'feature_indices': feature_indices
    }, savename_cls)
    
    '''test model'''
    X_test, Y_test, _ = load_data("test", feature_indices=feature_indices)
    test_dataset =  torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    acc, precision, recall, f1 = test(dataloader=test_loader, autoencoder = autoencoder, classifier = classifier)
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(acc, precision, recall, f1))
    
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")
