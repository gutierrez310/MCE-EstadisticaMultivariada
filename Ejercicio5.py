# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:28:21 2023

@author: carlo
"""

import numpy as np
from itertools import combinations

x = np.array([[-2., 1., 4.],
              [3., 0., -1.],
              [5., 1., 2],
              [-1., 3., 6.],
              [2., -7., 4.],
              [-1., 0., -1]])

n = x.shape[0]
p = x.shape[1]

# 6 observaciones de un vector aleatorio X = (X_1, X_2, X_3)

x_miu = sum(x)/n

x1 = x.copy()

print("X")
print(x)
for i in range(n):
    x1[i] -= x_miu

print("centralized X")
print(x1)

print("standarized X")
vars_x = np.var(x,axis=0)*n/(n-1)
for i in range(x.shape[0]):
    x1[i] /= vars_x
print(x1)

# Sx = E[(x2-mu_x2)(x3-mu_x3)]
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

# =============================================================================
# R = np.ones((p, p))
#
# for i in combinations(range(p), 2):
#     R[i[0], i[1]] = S[i[0], i[1]] / D[i[0], i[0]]**0.5 / D[i[1], i[1]]**0.5
#     R[i[1], i[0]] = S[i[0], i[1]] / D[i[0], i[0]]**0.5 / D[i[1], i[1]]**0.5
# =============================================================================
print("Y")
A = np.array([[-1, 2, -1],  # relaciones con X
              [1, 1, 0]])

Y = np.matmul(x, np.transpose(A))

print(Y)

times = np.matmul

print("covariance matrix of Y")
S_y = times(times(A, S), np.transpose(A))
print(S_y)

print("to get matriz diagonal D_y")
p = len(A)
D_y = np.zeros((p, p))
for i in range(p):
    D_y[i, i] = S_y[i, i]
print(D_y)

print("to get correlation matrix R_y")

D1 = np.linalg.inv(D_y**0.5)

R_y = np.matmul(np.matmul(D1, S_y), D1)
print(R_y)

print("Z")

B = np.array([[1/(6**0.5), 0],  # relaciones con Y
              [0, 1/(2**0.5)]])

Z = np.matmul(Y, np.transpose(B))

print(Z)

print("covariance matrix of Z")
S_z = times(times(B, S_y), np.transpose(B))
print(S_z)

print("to get matriz diagonal D_y")
p = len(B)
D_z = np.zeros((p, p))
for i in range(p):
    D_z[i, i] = S_z[i, i]
print(D_z)

print("to get correlation matrix R_z")

D1 = np.linalg.inv(D_z**0.5)

R_z = np.matmul(np.matmul(D1, S_z), D1)
print(R_z)






print("std X")

x1 = x.copy()
for i in range(n):
    x1[i] -= x_miu

for i in range(x.shape[0]):
    x1[i] /= vars_x

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


print(Z)

print("covariance matrix of Z")
S_z = times(times(B, S_y), np.transpose(B))
print(S_z)

print("to get matriz diagonal D_y")
p = len(B)
D_z = np.zeros((p, p))
for i in range(p):
    D_z[i, i] = S_z[i, i]
print(D_z)

print("to get correlation matrix R_z")

D1 = np.linalg.inv(D_z**0.5)

R_z = np.matmul(np.matmul(D1, S_z), D1)
print(R_z)










with open("T1.1-log.txt", "w") as f:
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
    f.write("\nY")
    f.write(str(Y))
    f.write("\ncovariance matrix of Y\n")
    f.write(str(S_y))
    f.write("\nto get matriz diagonal D_y\n")
    f.write(str(D_y))
    f.write("\nto get correlation matrix R_y\n")
    f.write(str(R_y))
    f.write("\nZ")
    f.write(str(Z))
    f.write("\ncovariance matrix of Z\n")
    f.write(str(S_z))
    f.write("\nto get matriz diagonal D_y\n")
    f.write(str(D_z))
    f.write("\nto get correlation matrix R_z\n")
    f.write(str(R_z))
