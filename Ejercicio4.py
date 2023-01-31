#  -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:19:01 2023

@author: carlo
"""


import pandas as pd
import seaborn as sns
import numpy as np
from itertools import combinations

df = pd.read_csv(
    'C:\\MCE 2nd\\Estadistica Multivariada\\Tareas\\Datos_tarea1.txt', sep='  '
    , header = None)
df.columns = ["a", "b", "c", "d", "e", "f", "g"]


# =============================================================================
# sns.pairplot(df[:])
# sns.heatmap(df.corr())
# sns.heatmap(df.corr(), cmap='PuOr')
# =============================================================================

x = np.array(df)
x = x.astype(np.float32)

n = x.shape[0]
p = x.shape[1]

# 42 observaciones de un vector aleatorio X = (X_1, X_2, X_3, ...)

x_miu = sum(x)/n

x1 = x.copy()

print("X")
print(x)
for i in range(n):
    x1[i] -= x_miu

print("centralized X")
print(x1)

# Sx = E[(x2-mu_x2)(x3-mu_x3)]
print("to get covariance matrix S")

S = np.zeros((p, p))

for i in range(p):
    S[i, i] = sum(x1[:, i]**2)/n

for i in combinations(range(p), 2):
    # centralized x1
    a = sum(list(map(lambda x: x[0]*x[1], x1[:, [i[0], i[1]]])))/n
    # print(i)
    # print(a)
    S[i[0], i[1]] = a
    S[i[1], i[0]] = a

print(S)

print("to get matriz diagonal D")

D = np.zeros((p, p))
for i in range(p):
    D[i, i] = S[i, i]
print(D)

print("to get correlation matrix R")

D1 = np.linalg.inv(D**0.5)

R = np.matmul(np.matmul(D1, S), D1)
print(R)

with open("T1-log.txt", "w") as f:
    f.write("X\n")
    f.write(str(x))
    f.write("\ncentralized X\n")
    f.write(str(x1))
    f.write("\nto get covariance matrix S\n")
    f.write(str(S))
    f.write("\nto get matriz diagonal D\n")
    f.write(str(D))
    f.write("\nto get correlation matrix R\n")
    f.write(str(R))
