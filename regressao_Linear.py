import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matplotlib inline

df = pd.read_csv('admissao.csv')
df.head()

positive = df[df['Admitido'].isin([1])]
negative = df[df['Admitido'].isin([0])]

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(positive['Exame1'], positive['Exame2'], s=50, c='b', marker='o', label='Admitido')
ax.scatter(negative['Exame1'], negative['Exame2'], s=50, c='r', marker='x', label='Não Admitido')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

# set X (training data) and y (target variable)
n_features = len(df.columns)-1

X = np.array(df.drop('Admitido',1))
y = df.iloc[:,n_features:n_features+1].values

mean = X.mean(axis=0)
std = X.std(axis=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)

def insert_ones(X):
    ones = np.ones([X.shape[0],1])
    return np.concatenate((ones,X),axis=1)

w = np.random.rand(1,n_features+1)

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(nums, sigmoid(nums), 'r')

def binary_cross_entropy(w, X, y):
    m = len(X)
    parte1 = np.multiply(-y, np.log(sigmoid(X @ w.T)))
    parte2 = np.multiply((1 - y), np.log(1 - sigmoid(X @ w.T)))

    somatorio = np.sum(parte1 - parte2)
    return  somatorio/m

def gradient_descent(w,X,y,alpha,epoch):
    cost = np.zeros(epoch)
    for i in range(epoch):
        w = w - (alpha/len(X)) * np.sum((sigmoid(X @ w.T) - y)*X, axis=0)
        cost[i] = binary_cross_entropy(w, X, y)

    return w,cost

X = insert_ones(X)
alpha=0.01 # taxa de aprendizado
epoch = 10000

w, cost = gradient_descent(w, X, y, alpha, epoch)
fig, ax = plt.subplots()
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')
ax.set_title('Erro vs. Epoch')

################################################################
#Realizando predições
def predict(w, X, threshold=0.5):
    p = sigmoid(X @ w.T) >= threshold
    return (p.astype('int'))

estudante1 = np.array([[45,85]])
estudante1 = (estudante1 - mean)/std
estudante1 = insert_ones(estudante1)

sigmoid(estudante1@ w.T)
predict(w, estudante1)

estudante2 = np.array([[90, 90]])
estudante2 = (estudante2 - mean)/std
estudante2 = insert_ones(estudante2)

sigmoid(estudante2 @ w.T)
predict(w, estudante2)

estudante3 = np.array([[45, 45]])
estudante3 = (estudante3 - mean)/std
estudante3 = insert_ones(estudante3)

sigmoid(estudante3 @ w.T)
predict(w, estudante3)