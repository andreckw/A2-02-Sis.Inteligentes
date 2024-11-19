import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def funcaolinear(x):
    return np.clip(3 * x + 50, 0, 200).astype(int)


def funcaorelu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transformispaid(isPaidDf):
    isPaid = []

    # True = 1, False = 0
    for p in isPaidDf.values:
        if p[0]:
            isPaid.append(1)
        else:
            isPaid.append(0)
    isPaid = np.array(isPaid).transpose()
    return isPaid


def transformsubject(subjectDf):
    subject = []
    # Business Finance = 0, Graphic Design = 1, Musical Instruments = 2, Web Development = 3 
    for s in subjectDf.values:
        if s[0] == "Business Finance":
            subject.append(0)
        elif s[0] == "Graphic Design":
            subject.append(1)
        elif s[0] == "Musical Instruments":
            subject.append(2)
        else:
            subject.append(3)
    subject = np.array(subject).transpose()
    return subject


def transformlevel(levelDf):
    level = []
    # All Levels = 0, Beginner Level = 1, Expert Level = 2, Intermediate Level = 3
    for ld in levelDf.values:
        if ld[0] == "All Levels":
            level.append(0)
        elif ld[0] == "Beginner Level":
            level.append(1)
        elif ld[0] == "Expert Level":
            level.append(2)
        else:
            level.append(3)
    level = np.array(level).transpose()
    return level



df = pd.read_csv("udemy_courses_dataset.csv")

nEpocas = 100
q = 100
# taxa de aprendizado
eta = 0.05

# 4 neuronios de entrada (isPaid, subject, level, contentDuration)
m = 4
# 3 neuronios escondida
N = 3
# 1 neuronio de saída
L = 1

isPaid = transformispaid(df.iloc[:q, [3]])

subject = transformsubject(df.iloc[:q, [11]])

level = transformlevel(df.iloc[:q, [8]])

contentDuration = np.array(df.iloc[:q, [9]]).transpose()

d = np.array(df.iloc[:q, [4]]).transpose()[0]

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

X = np.vstack((isPaid, subject, level, contentDuration))


# TREINAMENTO

for i in range(nEpocas):
    
    for j in range(q):
        Xb = np.hstack((b, X[:, j]))
        
        o1 = funcaorelu(W1.dot(Xb))
        o1b = np.insert(o1, 0, b)

        o2 = np.tanh(W2.dot(o1b))
        o2b = np.insert(o2, 0, b)

        o3 = np.tanh(W3.dot(o2b))
        o3b = np.insert(o3, 0, b)

        Y = funcaolinear(W4.dot(o3b))

        e = d[j] - Y

        E[j] = (e.dot(e).transpose()) / 2
        
        delta4 = np.diag(e).dot((1 - Y*Y))
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
t = 50

isPaid_test = transformispaid(df.iloc[:t, [3]])
subject_test = transformsubject(df.iloc[:t, [11]])
level_test = transformlevel(df.iloc[:t, [8]])
contentDuration_test = np.array(df.iloc[:t, [9]]).transpose()

X_test = np.vstack((isPaid_test, subject_test, level_test, contentDuration_test))

d_test = np.array(df.iloc[:t, [4]]).transpose()[0]

Error_Test = np.zeros(t)
for i in range(t):
    Xb = np.hstack((b, X[:, j]))
    
    o1 = funcaorelu(W1.dot(Xb))
    o1b = np.insert(o1, 0, b)

    o2 = np.tanh(W2.dot(o1b))
    o2b = np.insert(o2, 0, b)

    o3 = np.tanh(W3.dot(o2b))
    o3b = np.insert(o3, 0, b)

    Y = funcaolinear(W4.dot(o3b))

    Error_Test[i] = d_test[i] - Y

print(Error_Test)
