# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:58 2026

@author: ANANTHU
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.interpolate as si
#import statsmodels.api as sm
import scipy.stats as ss
import scipy.stats.mstats as ssm
import geone as gn
import pickle


data_dir = Path("data")
collar = pd.read_csv("data/collar.csv")
assay  = pd.read_csv("data/assay.csv")

print(collar.columns.tolist())
print(assay.columns.tolist())
print(collar.shape)
print(assay.shape)

assay["LENGTH"] = assay["TO"] - assay["FROM"]
print(assay["LENGTH"].describe())

assay["LENGTH"] = assay["TO"] - assay["FROM"]
assay["NI_X_LEN"] = assay["NI_PCT"] * assay["LENGTH"]

sums = assay.groupby("BHID")[["NI_X_LEN", "LENGTH"]].sum()

sums["NI_MEAN"] = sums["NI_X_LEN"] / sums["LENGTH"]
avg_grade = sums["NI_MEAN"].reset_index()
print(avg_grade.head())

holes = avg_grade.merge(collar[["BHID", "XCOLLAR", "YCOLLAR"]], on="BHID")
print(holes.head())
print(holes.shape)

fig, ax = plt.subplots(figsize=(8, 7))
scat = ax.scatter(holes["XCOLLAR"], holes["YCOLLAR"], c=holes["NI_MEAN"])
fig.colorbar(scat, ax=ax, label="Mean Ni [%]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
plt.show()


print(holes["NI_MEAN"].describe())
print(holes.isna().sum())


fig, ax = plt.subplots(1, 1)
ax.hist(holes["NI_MEAN"], bins=30, color="lightblue", edgecolor="black")
ax.set_xlabel("mean Ni [%]")
ax.set_ylabel("counts")
fig.savefig("figures/drillhole_locations.png", dpi=150)
plt.show()



coords = holes[["XCOLLAR", "YCOLLAR"]].values
values = holes["NI_MEAN"].values

fig, ax = plt.subplots(1, 1, num="Experimental variogram")

hexp, gexp, cexp = gn.covModel.variogramExp1D(coords, values,
                                              ncla=20,
                                              make_plot=True,
                                              hmax=1000)
ax.set_ylim(0)
plt.show()


print(f"Data variance: {np.var(values):.3f} [%^2]")
model = [("nugget", {"w": 0.002}),
         ("spherical", {"w": 0.048, "r": 380})]

cov_model = gn.covModel.CovModel1D(elem=model, name="model")

with open("01_OK_variogram-model.pkl", "wb") as file:
    pickle.dump(cov_model, file)
    
    hmax = 1000

fig, ax = plt.subplots(1, 1, num="Variogram fit", figsize=(8, 6))
hexp, gexp, cexp = gn.covModel.variogramExp1D(coords, values,
                                              ncla=20,
                                              make_plot=True,
                                              hmax=hmax)

cov_model.plot_model(vario=True, label="variogram model", hmax=hmax)
ax.set_ylim(0)
plt.legend()
plt.show()

x = holes["XCOLLAR"].to_numpy()
y = holes["YCOLLAR"].to_numpy()

dx = dy = 25                # cell size [m]
ox = x.min() - 100          # grid origin, with a 100 m margin
oy = y.min() - 100
nx = int((x.max() - x.min() + 200) / dx)
ny = int((y.max() - y.min() + 200) / dy)

x_est, y_est = np.meshgrid(np.linspace(ox+0.5*dx, ox-0.5*dx+nx*dx, nx),
                           np.linspace(oy+0.5*dy, oy-0.5*dy+ny*dy, ny))

xy_est = np.stack((x_est.ravel(), y_est.ravel()), axis=-1)
print(nx, ny, xy_est.shape)




vu, vu_std = gn.covModel.krige(coords, values, xy_est, cov_model,
                               method="ordinary_kriging")
ni_est = vu.reshape(x_est.shape)
ni_std = vu_std.reshape(x_est.shape)

extent = (x_est.min(), x_est.max(), y_est.min(), y_est.max())

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.imshow(ni_est, origin="lower", extent=extent)
fig.colorbar(im, ax=ax, label="Ni [%]")
ax.scatter(x, y, marker="x", color="white", s=8)
ax.set_title("Ordinary kriging: mean Ni")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.imshow(ni_std, origin="lower", extent=extent)
fig.colorbar(im, ax=ax, label="Std [%]")
ax.scatter(x, y, marker="x", color="white", s=8)
ax.set_title("Kriging standard deviation")
fig.savefig("figures/kriging_map.png", dpi=150)
plt.show()

