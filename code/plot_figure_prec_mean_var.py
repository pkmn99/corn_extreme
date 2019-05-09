"""
Scatter plot of mean precipitation and is variability
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_prism_data import load_gs_climate


ppt_gs = load_gs_climate(var='ppt', rerun=False)

ppt_mean = ppt_gs.mean().to_frame()
ppt_std = ppt_gs.std(ddof=0).to_frame()

sns.set()
sns.set_style('ticks')

fig, ax = plt.subplots(1, figsize=(6,4.5))
f = sns.regplot(x=ppt_mean.values[:,0],y=ppt_std.values[:,0], ax=ax)
f.set_xlabel('Mean precipitation (mm)')
f.set_ylabel('Precipitation variablity (mm)')
f.set_ylim(0,250)
f.set_xlim(0,900)

plt.subplots_adjust(bottom=0.2)

plt.savefig('../figure/figure_prec_mean_var.pdf')

