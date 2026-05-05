# Mega Case Study - Make a Hybrid Deep Learning Model

# https://github.com/minthawzin1995/SL_USL_MLearning/tree/master

# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

X = sc.fit_transform(X)

# Training the SOM

from minisom import MiniSom

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

som.random_weights_init(X)

som.train_random(data = X, num_iteration = 100)

# Visualizing the results

from pylab import bone, pcolor, colorbar, plot, show, savefig

bone()

pcolor(som.distance_map().T)

colorbar()

markers = ['o', 's']

colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

savefig("./test.png")

# Finding the frauds

mappings = som.win_map(X)

print("mappings =", mappings)  # defaultdict(<class 'list'>

print("mappings.keys =", mappings.keys())  # (5,3)， (8,3) 。。。

# print("mappings[(5,3)] =", mappings[(5,3)])  # list

# print("mappings[(8,3)] =", mappings[(8,3)])

# frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)  # mappings[(5,3)], mappings[(8,3)])

frauds = np.array(mappings[next(iter(mappings.keys()))])  

print("frauds.shape =", frauds.shape)  # (5, 15) 

frauds = sc.inverse_transform(frauds)  

print("frauds[0] =", frauds[0])

# exit(0)

# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features

customers = dataset.iloc[:, 1:].values  

print("customers.shape =", customers.shape)  # (690, 15)

print("customers[0] =", customers[0])

# Creating the dependent variable

is_fraud = np.zeros(len(dataset))  

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set

classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the probabilities of frauds

y_pred = classifier.predict(customers)

print("y_pred1.shape =", y_pred.shape)  # (690, 1)

print("y_pred[0] =", y_pred[0])

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

print("y_pred2.shape =", y_pred.shape)  

print("y_pred[0] =", y_pred[0])

y_pred = y_pred[y_pred[:, 1].argsort()]  

print(y_pred)