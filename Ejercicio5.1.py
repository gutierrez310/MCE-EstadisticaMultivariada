# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:20:11 2023

@author: carlo
"""


import pandas as pd
import seaborn as sns
import numpy as np
from itertools import combinations

df = pd.read_csv('C:\\MCE 2nd\\datos_ejercicio_semana_1.csv', sep=','
    , header = None)
df.columns = ["a", "b", "c"]

sns.pairplot(df[:])

x = np.array(df)
x = x.astype(np.float32)

n = x.shape[0]
p = x.shape[1]

print("X")
print(x)

print("mean of X")
x_miu = sum(x)/n
print(x_miu)

print("to get covariance matrix S")

x1 = x.copy()
for i in range(n):
    x1[i] -= x_miu

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

valores_propios, vectores_propios = np.linalg.eig(S)

print("varianza total")
# =============================================================================
# res=0
# for i in range(p):
#     res+=S[i,i]
# print(res)
# =============================================================================
print(total_var := sum(valores_propios))

print("varianza generalizada")
# =============================================================================
# res=1
# for i in valores_propios:
#     res *= i
# print(res)
# =============================================================================
print(gen_var := np.linalg.det(S))

with open("T1.2-log.txt", "w") as f:
    f.write("X\n")
    f.write(str(x))
    f.write("\nmean of X\n")
    f.write(str(x_miu))
    f.write("\nto get covariance matrix S\n")
    f.write(str(S))
    f.write("\nto get matriz diagonal D\n")
    f.write(str(D))
    f.write("\nto get correlation matrix R\n")
    f.write(str(R))
    f.write("\nvarianza total\n")
    f.write(str(total_var))
    f.write("\nvarianza generalizada\n")
    f.write(str(gen_var))


sns.heatmap(df.corr(), cmap='PuOr')