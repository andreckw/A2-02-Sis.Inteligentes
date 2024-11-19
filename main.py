import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def funcaolinear(x):
    return np.clip(1 * x + 50, 0, 200).astype(int)


def funcaorelu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



df = pd.read_csv("banana_quality.csv")

nEpocas = 100
q = 200
# taxa de aprendizado
eta = 0.2

# neuronios de entrada
m = 7
# neuronios escondida
N = 3
# 1 neuronio de saída
L = 1

size = np.array(df.iloc[:q, [0]]).transpose()[0]
weight = np.array(df.iloc[:q, [1]]).transpose()[0]
sweetnes = np.array(df.iloc[:q, [2]]).transpose()[0]
softnes = np.array(df.iloc[:q, [3]]).transpose()[0]
harvest = np.array(df.iloc[:q, [4]]).transpose()[0]
ripness = np.array(df.iloc[:q, [3]]).transpose()[0]
acidity = np.array(df.iloc[:q, [6]]).transpose()[0]

a = np.array(df.iloc[:q, [7]]).transpose()[0]
d = []
for k in a:
    if (k == "Good"):
        d.append(-1)
    else:
        d.append(1)

# bias
b = 1
# Matriz de entrada
W1 = np.random.random((N, m + 1))
# Matriz escondidas
W2 = np.random.random((N, N + 1))
W3 = np.random.random((N, N + 1))
# Matriz de saída
W4 = np.random.random((L, N + 1))

E = np.zeros(q)
etm = np.zeros(nEpocas)

X = np.vstack((size, weight, sweetnes, softnes, harvest, ripness, acidity))


# TREINAMENTO

for i in range(nEpocas):
    
    for j in range(q):
        Xb = np.hstack((b, X[:, j]))
        
        o1 = np.tanh(W1.dot(Xb))
        o1b = np.insert(o1, 0, b)

        o2 = sigmoid(W2.dot(o1b))
        o2b = np.insert(o2, 0, b)

        o3 = sigmoid(W3.dot(o2b))
        o3b = np.insert(o3, 0, b)
        Y = sigmoid(W4.dot(o3b))

        e = d[j] - Y
        print(d[j], Y)
        E[j] = (e.dot(e).transpose()) / 2
        
        delta4 = np.diag(e).dot((1 - Y * Y))
        vdelta4 = (W4.transpose()).dot(delta4)

        delta3 = np.diag(1 - o3b * o3b).dot(vdelta4)
        vdelta3 = (W3.transpose()).dot(delta3[1:])

        delta2 = np.diag(1 - o2b * o2b).dot(vdelta3)
        vdelta2 = (W2.transpose()).dot(delta2[1:])
        
        delta1 = np.diag(1 - o1b * o1b).dot(vdelta2)

        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2[1:], o1b))
        W3 = W3 + eta * (np.outer(delta3[1:], o2b))
        W4 = W4 + eta * (np.outer(delta4, o3b))
        

    etm[i] = E.mean()
    
plt.xlabel("Epocas")
plt.ylabel("Erro medio")
plt.plot(etm, color='c')
plt.plot(etm)
plt.show()


# TESTE

Error_Test = np.zeros(q)
for i in range(q):
    Xb = np.hstack((b, X[:, i]))
        
    o1 = np.tanh(W1.dot(Xb))
    o1b = np.insert(o1, 0, b)

    o2 = sigmoid(W2.dot(o1b))
    o2b = np.insert(o2, 0, b)

    o3 = sigmoid(W3.dot(o2b))
    o3b = np.insert(o3, 0, b)
    Y = sigmoid(W4.dot(o3b))

    Error_Test[i] = d[i] - Y

print(Error_Test)
print(np.round(Error_Test) - d)
