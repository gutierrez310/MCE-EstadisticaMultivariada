import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sts
import pingouin as pg
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\MCE 2nd\\Estadistica Multivariada\\Tareas\\Tarea5\\T1-5.dat',
                 sep=',', header=None)

df.columns = ["wind", "solar_rad", "CO", "NO", "NO2", "O3", "HC"]


# =============================================================================
# Realice un an ́alisis de regresi ́on utilizando solamente la primera respuestas Y1
# =============================================================================

X = df[["wind", "solar_rad"]]
Y = df[["NO2"]]  # df[["NO2", "O3"]]

n, r = X.shape
X = sm.add_constant(X, prepend=True)
A = np.linalg.inv(X.T@X)
beta = np.array(A@X.T@Y)

print("Fitted equation")
print(f"y_hat = {str(beta[0])} + {str(beta[1])}z_1 + {str(beta[2])}z_2")

err_hat = Y.to_numpy() - np.matmul(X.to_numpy(), beta)
s2 = np.dot(err_hat.T, err_hat)/(n-r-1)
var_beta = s2*np.diagonal(A)
var_beta = var_beta[0]

SS_res_X = np.dot(err_hat.T, err_hat)

print(beta)

# =============================================================================
# Analisis de Residuales
# =============================================================================


def unlist(x):
    new_list = []
    for i in range(len(x)):
        new_list.append(x[i][0])
    return new_list


err_hat = unlist(err_hat)
Y_pred = np.matmul(X.to_numpy(), beta)
Y_pred = unlist(Y_pred)

print(pg.normality(pd.DataFrame(err_hat)))

# 1. plot residuals vs predicted vals, no pattern visible
plt.scatter(err_hat, Y_pred)
plt.show()

# 1.a) residuals and predicted values independence
this_y = (Y_pred - np.mean(Y_pred))/np.var(Y_pred)
sts.chisquare(err_hat, this_y)  # indep

# 3. normal qq plot
ax = pg.qqplot(pd.DataFrame(err_hat), dist='norm')


# =============================================================================
# Realice un an ́alisis de regresi ́on multivariado utilizando ambas
# respuestas Y1 y Y2
# =============================================================================


X = df[["wind", "solar_rad"]]
Y = df[["NO2", "O3"]]

n, r = X.shape
X = sm.add_constant(X, prepend=True)
A = np.linalg.inv(X.T@X)
beta = np.array(A@X.T@Y)

print(beta)

err_hat = Y.to_numpy() - np.matmul(X.to_numpy(), beta)

Y_pred = np.matmul(X.to_numpy(), beta)

SS_err = err_hat.T@err_hat
# SS_pred = Y.T@Y - SS_err

sigma_hat = SS_err/(n-r-1)


# =============================================================================
# Residuales
# =============================================================================


cov_err = np.cov(err_hat.T)
print("La covarianza deberia ser de 0 pero es de", cov_err[0][1])


# no es buen modelos
print(pg.normality(pd.DataFrame(err_hat)))

