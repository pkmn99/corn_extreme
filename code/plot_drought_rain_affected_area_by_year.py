import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Modify the generated legend
"""
def custimize_legend(ax, label_text):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,label_text)

bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})

ts_drought_area = bin_yield.groupby(['Year','Prec_sigma_bin']).sum()\
    .loc[(slice(None), [2,3]), :]['Area'].unstack()
ts_rain_area = bin_yield.groupby(['Year','Prec_sigma_bin']).sum()\
    .loc[(slice(None), [13,14,15]), :]['Area'].unstack()

fig, [ax1,ax2] = plt.subplots(2,1, figsize=(8,10))

(ts_drought_area[[2,3]]/1000000*0.404686).plot.bar(edgecolor='k',stacked=True,ax=ax1,colors=sns.color_palette("RdBu", 7))
ax1.set_title('Maize area affected by extreme drought',fontsize=14)
ax1.set_ylabel('Area (million ha)',fontsize=14)
ax1.text(-0.1, 1, 'a', fontsize=16, transform=ax1.transAxes, fontweight='bold')

custimize_legend(ax1, ['<-2.5$\sigma$','(-2.5$\sigma$, -2.0$\sigma$)'])

order = [15,14,13]
(ts_rain_area[order]/1000000*0.404686).plot.bar(edgecolor='k',stacked=True,ax=ax2,colors=sns.color_palette("RdBu_r", 7))
ax2.set_title('Maize area affected by excessive rainfall',fontsize=14)
ax2.set_ylabel('Area (million ha)',fontsize=14)
ax2.text(-0.1, 1, 'b', fontsize=16, transform=ax2.transAxes, fontweight='bold')

custimize_legend(ax2, ['>3.5$\sigma$','(3.0$\sigma$, 3.5$\sigma$)','(2.5$\sigma$, 3.0$\sigma$)'])

plt.subplots_adjust(bottom = 0.1, top=0.95,hspace=0.25)

plt.savefig('../figure/figure_drought_rain_affected_area_by_year.pdf')

