import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sts
import pingouin as pg
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\MCE 2nd\\Estadistica Multivariada\\Tareas\\Tarea5\\P1-4.dat',
                 sep=',', header=None)

df.columns = ["sales", "profits", "assets"]

X = df[["sales", "assets"]]
Y = df[["profits"]]


# =============================================================================
# Ajuste un modelo de regresión lineal a los datos, considerando profits como
# la variable dependiente y sales y assets como la variable independiente
# =============================================================================

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


# =============================================================================
# Determine los intervalos de confianza simultáneos e individuales para un
# nivel de significancia de $\alpha=0.05$
# =============================================================================

alpha = 0.05
this_f = sts.f.ppf(1-alpha, r+1, n-r-1)**.5
SimCI_beta = []
print()
for i in range(r+1):
    SimCI_beta.append([beta[i][0]+np.sqrt(var_beta[i]*(r+1)*this_f),
                       beta[i][0]-np.sqrt(var_beta[i]*(r+1)*this_f)])
    print("CI. beta"+str(i))
    print(SimCI_beta[i])


# =============================================================================
# Aplique la prueba de la razón de verosimilitud para $H_{0}:\beta_{2}=0$
# con un nivel de significancia de $\alpha=0.05$. ¿El modelo debería
# ser modificado?
# =============================================================================

q = 1
err_hat1 = Y.to_numpy() - np.matmul(X.to_numpy()[:, :(q+1)], beta[q:])
SS_res_X1 = np.dot(err_hat1.T, err_hat1)
SS_res_X = np.dot(err_hat.T, err_hat)

likelihood_ratio_test = (SS_res_X1 - SS_res_X)/(r-q)/s2
likelihood_ratio_test = likelihood_ratio_test[0][0]
this_f = sts.f.ppf(1-alpha, r-q, n-r-1)
print("El efecto de beta_2 es significativo:", likelihood_ratio_test > this_f)


# =============================================================================
# Analice los residuales y discuta si el modelo es adecuado. Calcule los
# $leverage$ points asociados. ¿Algunas de estas compañías pueden considerarse
# datos atípicos en el conjunto de las variables explicativas?
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

this_Y = (Y_pred - np.mean(Y_pred))/np.var(Y_pred)
print("Normality in Y-pred")
print(pg.normality(pd.DataFrame(this_Y)))

sts.chisquare(err_hat, this_Y)  # indep

# 1.b) not constant variance
# See plot for err_hat vs Y_pred

# 3. normal qq plot
ax = pg.qqplot(pd.DataFrame(err_hat), dist='norm')
plt.show()


this_Y = unlist(Y.to_numpy())

H = X.to_numpy()@A@X.T.to_numpy()

leverages = np.diag(H)
print()
print("leverages")
print(leverages)

model = sm.OLS(Y, X).fit()
influence = model.get_influence()
cooks = influence.cooks_distance  # low influence throughout

D1 = []
s2 = SS_res_X / (n-r)
s2 = s2[0][0]
for i in range(n):
    D1.append(err_hat[i]**2*leverages[i]/r/s2/(1-leverages[i])**2)
influential_obs = D1 > sts.f.ppf(.5, r, n-r)
print()
print("D1, distancias de cook")
print(D1)

print("Highest influential obs")
print(df[influential_obs])

print("No outliers, imo")
# http://www.csam.or.kr/journal/view.html?doi=10.5351/CSAM.2017.24.3.317


# =============================================================================
# Obtenga el intervalo de predicción del $95 \%$ correspondiente
# a $sales=100$ y $assets=500$
# =============================================================================

to_pred = np.array([1, 100, 500])

this_pred = np.matmul(to_pred.reshape((1, 3)), SimCI_beta)[0]

this_t = sts.t.ppf(1-alpha/2, n-r-1)
this_dif = this_t*np.sqrt(s2*(1+to_pred.T@A@to_pred))
this_pred = to_pred@beta
pred_CI = [to_pred@beta+this_dif, to_pred@beta-this_dif]
