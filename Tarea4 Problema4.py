import pandas as pd
import numpy as np

x1 = pd.DataFrame({"x1": [6, 5, 8, 4, 7], "x2": [7, 9, 6, 9, 9]})
x2 = pd.DataFrame({"x1": [3, 1, 2], "x2": [3, 6, 3]})
x3 = pd.DataFrame({"x1": [2, 5, 3, 2], "x2": [3, 1, 1, 3]})

p = 2
g = 3

def Mean_Treatment_Residual(tuple_of_treatments):
    g = len(tuple_of_treatments)  # number of treatments
    everything = pd.concat(tuple_of_treatments)
    global_means = sum(everything.to_numpy())/len(everything)

    lens_of_treatments = []
    means_of_treatments = []
    for i in range(g):
        lens_of_treatments.append(len(tuple_of_treatments[i]))
        means_of_treatments.append(sum(tuple_of_treatments[i].to_numpy())/lens_of_treatments[i])

    W = 0
    for i in range(g):
        W += (lens_of_treatments[i]-1)*np.cov(tuple_of_treatments[i].T)

    B_and_W = np.cov(everything.T)*(len(everything)-1)

    B = B_and_W - W

    print()
    print("Medias de los tratamientos")
    print(means_of_treatments)

    print()
    print("Longitudes de los tratamientos")
    print(lens_of_treatments)
    print(global_means)
    return global_means, B, W

M, B, W = Mean_Treatment_Residual((x1, x2, x3))
print()
print("Medias globales")
print(M)

wilks_lambda = np.linalg.det(W)/np.linalg.det(B+W)


alpha = 0.01
import scipy.stats as sts
print()
print("Valor critico: ")
print(sts.f.ppf(1-alpha, 2*p, 2*(12-p-2)))



