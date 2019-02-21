import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_figure1 import define_colors


bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})


colors = define_colors()


x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
x_txt.insert(0,'')
x_txt.append('')

# Total county sample number, n
n=bin_yield.groupby('Prec_sigma_bin').count()['Year'].sum()

bar_count = (bin_yield.groupby('Prec_sigma_bin').count()['Year']/n*100)


fig, ax = plt.subplots(1,1, figsize=(8,6))

bar_count.plot.bar(color=colors,width=0.75, ax=ax)

ax.set_xticks(np.arange(-0.5,14.5,1))
ax.set_xlim(-0.5,13.5)
ax.set_xticklabels(x_txt, rotation=0)


ax.set_ylabel('County sample (%)', fontsize=12)

ax.set_xlabel('Precipitation anomaly ($\sigma$)',labelpad=17, fontsize=12)
ax.text(0.0, -0.075, 'Extreme dry', transform=ax.transAxes, fontsize=10,
               color=colors[0])
ax.text(0.25, -0.075, 'Moderate dry', transform=ax.transAxes, fontsize=10,
               color=colors[2])
ax.text(0.5, -0.075, 'Moderate wet', transform=ax.transAxes, fontsize=10,
               color=colors[6])
ax.text(0.8, -0.075, 'Extreme wet', transform=ax.transAxes, fontsize=10,
               color=colors[-1])

ax.text(0.8, 0.9, '$n$=%d'%n, transform=ax.transAxes, fontsize=12,
               color='k')

# Change xlabel color
[t.set_color(colors[0]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i<3]
[t.set_color(colors[-1]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i>=11]

plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.3)

plt.savefig('../figure/figure_prec_sigma_sample_number.pdf')

