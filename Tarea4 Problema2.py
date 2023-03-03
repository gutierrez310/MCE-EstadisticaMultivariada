import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

S = np.array([[3266.46, 1343.97, 731.54, 1175.50, 162.68, 238.37],
              [1343.97, 721.91, 324.25, 537.35, 80.17, 117.73],
              [731.54, 324.25, 179.28, 281.17, 39.15, 56.80],
              [1175.50, 537.35, 281.17, 474.98, 63.73, 94.85],
              [162.68, 80.17, 39.15, 63.73, 9.95, 13.88],
              [238.37, 117.73, 56.80, 94.85, 13.88, 21.26]])
x_bar = np.array([95.52, 164.38, 55.69, 93.39, 17.98, 31.13])


n = 61
p = 6
alpha = 0.05
SimICs = []
for i in range(p):
    diff = ((n-1)*p*sts.f.ppf(1 - alpha, p, n-p)/n/(n-p)*S[i, i])**0.5
    SimICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza Simultaneos")
print(SimICs)


def confidence_ellipse(means, S, ax, varss=(0, 3),
                       n_std=3.0, facecolor='none', **kwargs):
    mean_y = means[varss[1]]
    var_y = S[varss[1], varss[1]]
    mean_x = means[varss[0]]
    var_x = S[varss[0], varss[0]]
    covv = S[varss[0], varss[1]]

    pearson = covv/(var_x*var_y)**0.5

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x, height=ell_radius_y,
                      facecolor='none', edgecolor='red')

    scale_x = (var_x**0.5)# * n_std

    scale_y = (var_y**0.5)# * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# 3 es girth es vertical
# 0 es weight es horizontal
fig, ax = plt.subplots()
ax.set_title("Ellipsoid for alpha="+str(alpha)+", x=Weight, y=Girth")
std = sts.chi2.ppf(1-alpha, df=2)/2
confidence_ellipse(x_bar, S, ax, varss=(0, 3), edgecolor='red', n_std=std)

plt.axhline(y=SimICs[3][0], color="grey", linestyle="--")
plt.axhline(y=SimICs[3][1], color="grey", linestyle="--")
plt.axvline(x=SimICs[0][0], color="grey", linestyle="--")
plt.axvline(x=SimICs[0][1], color="grey", linestyle="--")
mean_x = x_bar[0]
mean_y = x_bar[3]
plt.show()




BonferroniICs = []
for i in range(p):
    diff = sts.t.ppf(1-alpha/p/2, n-1)*((S[i, i]/n)**0.5)
    BonferroniICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza de Bonferroni")
print(BonferroniICs)

fig, ax = plt.subplots()
ax.set_title("Ellipsoid for alpha="+str(alpha)+", x=Weight, y=Girth")
std = sts.chi2.ppf(1-alpha, df=2)/2
confidence_ellipse(x_bar, S, ax, varss=(0, 3), edgecolor='red', n_std=std)

plt.axhline(y=SimICs[3][0], color="green", linestyle="--")
plt.axhline(y=SimICs[3][1], color="green", linestyle="--")
plt.axvline(x=SimICs[0][0], color="green", linestyle="--")
plt.axvline(x=SimICs[0][1], color="green", linestyle="--")

plt.axhline(y=BonferroniICs[3][0], color="blue", linestyle="-.")
plt.axhline(y=BonferroniICs[3][1], color="blue", linestyle="-.")
plt.axvline(x=BonferroniICs[0][0], color="blue", linestyle="-.")
plt.axvline(x=BonferroniICs[0][1], color="blue", linestyle="-.")

plt.show()









varss = (5, 4)
this_var = S[varss[0], varss[0]] + S[varss[1], varss[1]] - 2*S[varss[1], varss[0]]
this_mean = x_bar[varss[0]] - x_bar[varss[1]]
diff = sts.t.ppf(1-alpha/7/2, n-1)*((this_var/n)**0.5)
BonferroniICs.append((this_mean+diff, this_mean-diff))
print("Intervalo de confianza para la diferencia de media de Head width y media de Head length")
print(BonferroniICs[-1])









BIGSimICs = []
this_chi2 = sts.chi2.ppf(1-alpha, p)
for i in range(p):
    diff = (this_chi2*S[i, i]/n)**0.5
    BIGSimICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza Simultaneos, tomando en cuenta la aproximacion para muestras grandes")
print(BIGSimICs)

fig, ax = plt.subplots()
ax.set_title("Ellipsoid for alpha="+str(alpha)+", x=Weight, y=Girth")
std = sts.chi2.ppf(1-alpha, df=2)/2
confidence_ellipse(x_bar, S, ax, varss=(0, 3), edgecolor='red', n_std=std)

plt.axhline(y=BIGSimICs[3][0], color="grey", linestyle="--")
plt.axhline(y=BIGSimICs[3][1], color="grey", linestyle="--")
plt.axvline(x=BIGSimICs[0][0], color="grey", linestyle="--")
plt.axvline(x=BIGSimICs[0][1], color="grey", linestyle="--")
mean_x = x_bar[0]
mean_y = x_bar[3]
plt.show()


BIGBonferroniICs = []
this_z = sts.norm.ppf(1-alpha/2/p)
for i in range(p):
    diff = this_z*((S[i, i]/n)**0.5)
    BIGBonferroniICs.append((x_bar[i]+diff, x_bar[i]-diff))
print("Intervalos de Confianza de Bonferroni, tomando en cuenta la aproximacion para muestras grandes")
print(BIGBonferroniICs)


fig, ax = plt.subplots()
ax.set_title("Ellipsoid for alpha="+str(alpha)+", x=Weight, y=Girth (BIG)")
std = sts.chi2.ppf(1-alpha, df=2)/2
confidence_ellipse(x_bar, S, ax, varss=(0, 3), edgecolor='red', n_std=std)

plt.axhline(y=BIGSimICs[3][0], color="green", linestyle="--")
plt.axhline(y=BIGSimICs[3][1], color="green", linestyle="--")
plt.axvline(x=BIGSimICs[0][0], color="green", linestyle="--")
plt.axvline(x=BIGSimICs[0][1], color="green", linestyle="--")

plt.axhline(y=BIGBonferroniICs[3][0], color="blue", linestyle="-.")
plt.axhline(y=BIGBonferroniICs[3][1], color="blue", linestyle="-.")
plt.axvline(x=BIGBonferroniICs[0][0], color="blue", linestyle="-.")
plt.axvline(x=BIGBonferroniICs[0][1], color="blue", linestyle="-.")

plt.show()


varss = (5, 4)
this_var = S[varss[0], varss[0]] + S[varss[1], varss[1]] - 2*S[varss[1], varss[0]]
this_mean = x_bar[varss[0]] - x_bar[varss[1]]
diff = sts.norm.ppf(1-alpha/7/2)*((this_var/n)**0.5)
BIGBonferroniICs.append((this_mean+diff, this_mean-diff))
print("Intervalo de confianza para la diferencia de media de Head width y media de Head length, tomando en cuenta la aproximacion para muestras grandes")
print(BIGBonferroniICs[-1])
