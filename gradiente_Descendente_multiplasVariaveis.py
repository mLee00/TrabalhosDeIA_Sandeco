import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

%matplotlib inline

df = pd.read_csv('prices.csv')
df.head()

df_norm = (df - df.mean()) / df.std()
df_norm.head()

n_features = len(df_norm.columns)-1
n_features

w = np.random.rand(1,n_features+1)

X = np.array(df_norm.drop('price', axis=1))
y = df_norm.iloc[:, 2:3].values

def insert_ones(X):
    ones = np.ones([X.shape[0],1])
    return np.concatenate((ones,X) , axis=1)

def custo(w,X,y):
    m = len(X)
    erro = (X@w.T - y)
    custo = np.sum(np.power(erro,2))
    return custo/m

def gradient_descent(w, X, y, alpha, epoch):
    cost = np.zeros(epoch)
    for i in range(epoch):
        w = w - (alpha/len(X)) * np.sum((X@w.T - y)*X, axis=0)
        cost[i] = custo(w, X, y)
    return w, cost

X = insert_ones(X)
w, cost = gradient_descent(w, X, y, alpha, epoch)
custo(w, X, y)

#Plot do custo
fig, ax = plt.subplots()
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')
ax.set_title('Erro vs. Epoch')

print(w)