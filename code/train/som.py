import shutup  

shutup.please()
import numpy as np
from torch.autograd import Variable

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import numpy as np

from torch.autograd import Variable

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

import pylab as pl

import math

import numpy as np

# from minisom import MiniSom

from utils.minisom import MiniSom

from sklearn import datasets

from numpy import sum as npsum

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib.gridspec import GridSpec

import pandas as pd

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder

import pickle

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score,classification_report,multilabel_confusion_matrix,hamming_loss

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import  StratifiedKFold

import itertools

import argparse

import pickle

from DataProcess import load_data,process

def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  
        coo_Y = []  
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)
    pl.legend(loc='upper right')
    pl.show()

def classify(som,data,winmap):
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show(som, output_size, result_path):
    """Plot label maps on the output layer."""
    plt.figure(figsize=(16, 16))
    all_markers = ['o', 's', 'D', 'v', 'P', '*', 'X']
    all_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    all_category_color = {'0': 'C0', '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6'}
    all_class_names = ['0', '1', '2', '3', '4', '5', '6']
    markers = all_markers[0:output_size] # 'V', 'P', '*', 'X'
    colors = all_colors[0:output_size]
    category_color = dict(itertools.islice(all_category_color.items(), output_size))
    class_names = all_class_names[0:output_size]
    heatmap = som.distance_map()
    plt.pcolor(heatmap, cmap='bone_r')
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)
        plt.plot(w[0] + .5, w[1] + .5, markers[Y_train[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[Y_train[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, size, 0, size])
    ax = plt.gca()
    ax.invert_yaxis()
    legend_elements = [Patch(facecolor=clr, edgecolor='w', label=l) for l, clr in category_color.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    # plt.show()
    # plt.savefig("D:\\wamp\\www\\multi_omics_own\\download_data\\Jobs\\"+jobID+"\\result\\figure"+"_1.png", format='png')
    plt.savefig(result_path + "figure.png", format='png', dpi=300)
    plt.close()

if __name__ == '__main__':
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
    output_size = args.output_size  
    type = args.type
    feature_method = args.feature_method if args.feature_method and args.feature_method.strip() else None
    print('model = SOM')
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
    print('feature selection method =', feature_method if feature_method else 'none')
    set_random_seed(random_seed)  
    if (ratio != "0"):
        process(ratio=ratio)
    iterations = epochs
    if (kfold):
        kfscore = []
       
        skf = StratifiedKFold(n_splits=kfold)  
        X_train_total, Y_train_total, feature_indices = load_data("train", feature_method=feature_method)
        actual_num_classes = len(torch.unique(torch.LongTensor(Y_train_total)))
        if actual_num_classes != output_size:
            print(f"Warning: The number of classes set by the user ({output_size}) does not match the actual number of classes in the data ({actual_num_classes})")
            print(f"Automatically use the actual number of classes: {actual_num_classes}")
            output_size = actual_num_classes
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train_total, Y_train_total)):
            print("--------The {} fold is training---------".format(i+1))
            X_train, X_val = np.array(X_train_total)[train_idx], np.array(X_train_total)[val_idx]
            Y_train, Y_val = np.array(Y_train_total)[train_idx], np.array(Y_train_total)[val_idx]
            N = X_train.shape[0]
            M = X_train.shape[1]
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            print("The best side length of the grid is :", size)
            som = MiniSom(size, size, M, sigma=3, learning_rate=lr, neighborhood_function='bubble')
            try:
                som.pca_weights_init(X_train)
            except ValueError as e:
                print(f"[SOM] pca_weights_init failed ({e}); fallback to random_weights_init.")
                som.random_weights_init(X_train)
            som.train_batch(X_train, iterations, verbose=False)
            winmap = som.labels_map(X_train, Y_train)
            print("Finish training!")  
            print("--------The {} fold validation result---------".format(i+1))
            y_pred = classify(som, X_val, winmap)
            acc = accuracy_score(Y_val, y_pred)
            precision, recall, f1 = precision_recall_fscore_support(Y_val, y_pred, average='weighted')[:-1]
            print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))
            kfscore.append([acc, precision, recall, f1])
        # average score
        kfscore = np.array(kfscore).sum(axis= 0)/float(kfold)  # acc, precision, recall, f1
        print("--------KFold Final Average Validation Results---------")
        print("Stratified KFold mean validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(kfscore[0], 4), round(kfscore[1], 4), round(kfscore[2], 4), round(kfscore[3], 4)))
    
    else:
        X_train, Y_train, feature_indices = load_data("train", feature_method=feature_method)
        actual_num_classes = len(torch.unique(torch.LongTensor(Y_train)))
        if actual_num_classes != output_size:
            print(f"警告：用户设置的类别数 ({output_size}) 与数据实际类别数 ({actual_num_classes}) 不一致")
            print(f"自动使用实际类别数：{actual_num_classes}")
            output_size = actual_num_classes
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_val, Y_val, _ = load_data("validate", feature_indices=feature_indices)
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        N = X_train.shape[0]
        M = X_train.shape[1]
        size = math.ceil(np.sqrt(5 * np.sqrt(N)))
        print("The best side length of the grid is :", size)
        som = MiniSom(size, size, M, sigma=3, learning_rate=lr, neighborhood_function='bubble')
        try:
            som.pca_weights_init(X_train)
        except ValueError as e:
            print(f"[SOM] pca_weights_init failed ({e}); fallback to random_weights_init.")
            som.random_weights_init(X_train)
        som.train_batch(X_train, iterations, verbose=False)
        winmap = som.labels_map(X_train, Y_train)
        print("Finish training!")  
        y_pred = classify(som, X_val, winmap)
        acc = accuracy_score(Y_val, y_pred)
        precision, recall, f1 = precision_recall_fscore_support(Y_val, y_pred, average='weighted')[:-1]
        print("validation acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))
    if feature_indices is not None:
        print("selected feature indices (1-based):", feature_indices + 1)
    else:
        print("feature selection: none (all features)")
    print("Done!")  

    result_path = "./result/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savename = result_path+'model.pth'

    checkpoint = {
        'som': som,
        'winmap': winmap,
        'feature_indices': feature_indices,
        'output_size': output_size,
        'learning_rate': lr,
        'epochs': epochs,
        'kfold': kfold,
        'batch_size': batch_size,
    }
    torch.save(checkpoint, savename)

    show(som, output_size, result_path)
    X_test, Y_test, _ = load_data("test", feature_indices=feature_indices)
    y_pred = classify(som, X_test, winmap)
    acc = accuracy_score(Y_test,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(Y_test, y_pred, average='weighted')[:-1]
    print("test acc = {}, precision = {}, recall = {}, f1 = {}".format(round(acc,4), round(precision,4), round(recall,4), round(f1,4)))
    
    with open(result_path + "test_result.txt", mode="w") as f:
        f.write("test result: \n")
        f.write("acc = " + str(round(acc, 4)) + "\n")
        f.write("precision = " + str(round(precision, 4)) + "\n")
        f.write("recall = " + str(round(recall, 4)) + "\n")
        f.write("f1 = " + str(round(f1, 4)) + "\n")
    print('test set result')
    print(classification_report(Y_test, np.array(y_pred)))  
