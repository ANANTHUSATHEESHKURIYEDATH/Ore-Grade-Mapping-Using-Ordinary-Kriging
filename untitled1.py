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



