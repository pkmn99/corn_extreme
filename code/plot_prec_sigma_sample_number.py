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
n = bin_yield.groupby('Prec_sigma_bin').count()['Year'].sum()
bar_count = (bin_yield.groupby('Prec_sigma_bin').count()['Year']/n*100)

# Negative yeild percentage
c0 = bin_yield['Yield_ana_to_yield']<0
p_neg = 100 * bin_yield.loc[c0,:].groupby('Prec_sigma_bin').count()['Year'] \
              /bin_yield.groupby('Prec_sigma_bin').count()['Year']

# Begin plot
fig, [ax,ax2] = plt.subplots(2,1, figsize=(8,12))

bar_count.plot.bar(color=colors,width=0.75, ax=ax,edgecolor='k')

ax.set_xticks(np.arange(-0.5,14.5,1))
ax.set_xlim(-0.5,13.5)
ax.set_xticklabels(x_txt, rotation=0)

ax.set_yticks(np.arange(0,21,5))
ax.set_ylabel('County sample (%)', fontsize=12)

ax.set_xlabel('Precipitation anomaly ($\sigma$)',labelpad=17, fontsize=12)
ax.text(0.0, -0.09, 'Extreme dry', transform=ax.transAxes, fontsize=10,
               color=colors[0])
ax.text(0.25, -0.09, 'Moderate dry', transform=ax.transAxes, fontsize=10,
               color=colors[2])
ax.text(0.5, -0.09, 'Moderate wet', transform=ax.transAxes, fontsize=10,
               color=colors[6])
ax.text(0.8, -0.09, 'Extreme wet', transform=ax.transAxes, fontsize=10,
               color=colors[-1])

ax.text(0.8, 0.9, '$n$=%d'%n, transform=ax.transAxes, fontsize=12,
               color='k')


# plot 2nd panel
p_neg.plot(color='k',legend=False,ax=ax2,marker='.')
ax2.set_xlabel('Precipitation anomaly ($\sigma$)',labelpad=17, fontsize=12)
ax2.set_ylabel('Percentage of negative yield impact (%)', fontsize=12)

ax2.set_xlim([1.5,15.5])
ax2.set(xticks=np.arange(1.5,15.5,1), xticklabels=(x_txt))

ax2.text(0.0, -0.09, 'Extreme dry', transform=ax2.transAxes, fontsize=10,
               color=colors[0])
ax2.text(0.25, -0.09, 'Moderate dry', transform=ax2.transAxes, fontsize=10,
               color=colors[2])
ax2.text(0.5, -0.09, 'Moderate wet', transform=ax2.transAxes, fontsize=10,
               color=colors[6])
ax2.text(0.8, -0.09, 'Extreme wet', transform=ax2.transAxes, fontsize=10,
                   color=colors[-1])

# Change xlabel color
[t.set_color(colors[0]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i<3]
[t.set_color(colors[-1]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i>=11]

[t.set_color(colors[0]) for i,t in enumerate(ax2.xaxis.get_ticklabels()) if i<3]
[t.set_color(colors[-1]) for i,t in enumerate(ax2.xaxis.get_ticklabels()) if i>=11]

ax.text(-0.1, 1, 'a', fontsize=16, transform=ax.transAxes, fontweight='bold')
ax2.text(-0.1, 1, 'b', fontsize=16, transform=ax2.transAxes, fontweight='bold')

plt.subplots_adjust(top=0.9, bottom=0.15)

plt.savefig('../figure/figure_prec_sigma_sample_number.pdf')

