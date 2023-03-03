import scipy.stats as sts
import pandas as pd
import numpy as np

df = pd.read_csv(
    'C:\\MCE 2nd\\Estadistica Multivariada\\Tareas\\T6-9.txt', sep='  ',
    header=None)

df_female1 = df[:24]
df_female = pd.DataFrame()
df_female["x1"] = df_female1[0]
df_female["x2"] = df_female1[1]
df_female["x3"] = df_female1[2]

df_male1 = df[24:]
df_male = pd.DataFrame()
df_male["x1"] = df_male1[0]
df_male["x2"] = df_male1[1]
df_male["x3"] = df_male1[2]

n1 = len(df_female)
n2 = len(df_male)

p = len(df_female.iloc[0])

# =============================================================================
#
# =============================================================================

# a)

alpha = 0.05
x1 = np.array(df_female)
x2 = np.array(df_male)

x1_miu = sum(x1)/p
x2_miu = sum(x2)/p

S1 = np.cov(x1.T)
S2 = np.cov(x2.T)

S_pooled = S1/2+S2/2  # n1=n2   = (n1-1)/(n1+n2-2)S1 + (n2-1)/(n1+n2-2)S2

estadistico_de_prueba = lambda x1_miu, x2_miu, S_pooled: np.dot(np.matmul((x1_miu-x2_miu).T, np.linalg.inv(S_pooled)), (x1_miu-x2_miu))

print("Estadistico_de_prueba", estadistico_de_prueba(x1_miu, x2_miu, (1/n1+1/n2)*S_pooled))

crit_val = (n1+n2-2)*p/(n1+n2-p-1)*sts.f.ppf(1-alpha, p, n1+n2-p-1)
print("Valor critico", crit_val)


# Muestra grande  n1-p = 21
print("Asumiendo que las muestras son 'grandes'")
S_pooled = (S1/n1 + S2/n2)
print("Estadistico_de_prueba", estadistico_de_prueba(x1_miu, x2_miu, S_pooled))

crit_val = sts.chi2.ppf(1-alpha, p)
print("Valor critico para muestras grandes", crit_val)


print()
print("Despues de aplicar una transformacion logaritmica, los datos son normales")
# Tal vez transformando los datos a m.normales
from pingouin import multivariate_normality
multivariate_normality(np.log(df_female), alpha=0.05)
multivariate_normality(np.log(df_male), alpha=0.05)
# son normales bajo la transformacion logaritmica

x1 = np.array(np.log(df_female))
x2 = np.array(np.log(df_male))

x1_miu = sum(x1)/p
x2_miu = sum(x2)/p

S1 = np.cov(x1.T)
S2 = np.cov(x2.T)

S_pooled = S1/2 + S2/2  # n1 = n2

print("Estadistico_de_prueba", estadistico_de_prueba(x1_miu, x2_miu, (1/n1+1/n2)*S_pooled))

crit_val = (n1+n2-2)*p/(n1+n2-p-1)*sts.f.ppf(1-alpha, p, n1+n2-p-1)
print("Valor critico", crit_val)


print("Asumiendo que las muestras son 'grandes'")
S_pooled = (S1/n1 + S2/n2)
print("Estadistico_de_prueba", estadistico_de_prueba(x1_miu, x2_miu, S_pooled))

crit_val = sts.chi2.ppf(1-alpha, p)
print("Valor critico para muestras grandes", crit_val)

print()
print("""Podemos observar que con o sin la suposicion de muestras grandes, el
      estadistico de prueba no supera el valor critico de la prueba. Esto es
      visible tambien bajo la transformacion logaritmica. Con nivel de signifi-
      cancia del 5%, rechazamos la hipotesis nula de que la diferencia de las
      medias entre las muestras es de 0.""")
print("""Vale mencionar que la transformacion logaritmica ayuda a reforzar los
      supuestos necesarios para la comparacion de medias, por lo que es
      preferible en este caso.""")
print()

# =============================================================================
#
# =============================================================================

# b) Combinacion lineal mas responsable

a_hat = np.matmul(np.linalg.inv(1/n1*S1+1/n2*S2), x1_miu-x2_miu)
print("La combinacion lineal de los componentes mas responsable por el rechazo de H_0.")
print(a_hat)
print()

# =============================================================================
#
# =============================================================================

# c) Intervalos de confianza

n = n1+n2-1
S_pooled = S1/2+S2/2  # n1=n2   = (n1-2)*S1/(n1+n2-2) + (n2-1)*S2/(n1+n2-2)
S = S_pooled
x_bar = x1_miu-x2_miu

# Intervalos Simultaneos
SimICs = []
this_t2 = (n-1)*p*sts.f.ppf(1 - alpha, p, n-p)/n/(n-p)
for i in range(p):
    diff = (this_t2*(2/n1)*S[i, i])**0.5  # n1=2   = 1/n1+1/n2
    SimICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza Simultaneos")
print(SimICs)

# Intervalos Simultaneos de Bonferroni
BonferroniICs = []
this_t = sts.t.ppf(1-alpha/p/2, n-1)
for i in range(p):
    diff = this_t*((2/n1)*(S[i, i]))**0.5  # n1=2   = 1/n1+1/n2
    BonferroniICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza de Bonferroni")
print(BonferroniICs)

print()
print("Estos intervalos son sobre la transformacion logaritmica")


print()
print("Sobre los datos originales")

alpha = 0.05
x1 = np.array(df_female)
x2 = np.array(df_male)

x1_miu = sum(x1)/p
x2_miu = sum(x2)/p

S1 = np.cov(x1.T)
S2 = np.cov(x2.T)

S_pooled = S1/2+S2/2  # n1=n2   = (n1-2)*S1/(n1+n2-2) + (n2-1)*S2/(n1+n2-2)
S = S_pooled
x_bar = x1_miu-x2_miu
# Intervalos Simultaneos
SimICs = []
this_t2 = (n-1)*p*sts.f.ppf(1 - alpha, p, n-p)/n/(n-p)
for i in range(p):
    diff = (this_t2*(2/n1)*S[i, i])**0.5  # n1=2   = 1/n1+1/n2
    SimICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza Simultaneos")
print(SimICs)

# Intervalos Simultaneos de Bonferroni
BonferroniICs = []
this_t = sts.t.ppf(1-alpha/p/2, n-1)
for i in range(p):
    diff = this_t*((2/n1)*(S[i, i]))**0.5  # n1=2   = 1/n1+1/n2
    BonferroniICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza de Bonferroni")
print(BonferroniICs)


print("""Puede que los intervalos de confianza contengan al '0' en el espacio
      original, pero ya que la prueba resulta en el rechazo de la hipotesis
      nula ya que esto demuestra que no en todas las combinacion lineales
      $a^T(\\mu_1-\\mu_2)$ se contiene al '0'.""")
