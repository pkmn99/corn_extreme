import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Plot the corn harvest area affected by drought and extreme rainfall during the study period

"""
Modify the generated legend
"""
def custimize_legend(ax, label_text):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,label_text)

bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})


fig, [ax1,ax2] = plt.subplots(1,2, figsize=(15,8))

c1 = bin_yield['Prec_sigma_bin']<4
(bin_yield[c1].groupby(['State','Prec_sigma_bin']).sum()['Area']/1000000*0.404686).unstack('Prec_sigma_bin') \
        .plot.bar(edgecolor='k',stacked='True',ax=ax1,color=sns.color_palette("RdBu", 7))
ax1.set_ylabel('Area (million ha)',fontsize=14)
ax1.set_title('Maize area affected by extreme drought',fontsize=14)
ax1.set_ylim([0,9.5*0.404686])
ax1.text(-0.1, 1, 'a', fontsize=16, transform=ax1.transAxes, fontweight='bold')

custimize_legend(ax1, ['<-2.5$\sigma$','(-2.5$\sigma$, -2.0$\sigma$)'])


order = [15,14,13]
c2 = bin_yield['Prec_sigma_bin']>12
(bin_yield[c2].groupby(['State','Prec_sigma_bin']).sum()['Area']/1000000*0.404686).unstack('Prec_sigma_bin')[order] \
        .plot.bar(edgecolor='k',stacked='True',ax=ax2,color=sns.color_palette("RdBu_r", 7))
ax2.set_ylabel('Area (million ha)',fontsize=14)
ax2.set_title('Maize area affected by excessive rainfall',fontsize=14)
ax2.set_ylim([0,9.5*0.404686])
ax2.text(-0.1, 1, 'b', fontsize=16, transform=ax2.transAxes, fontweight='bold')

custimize_legend(ax2, ['>3.5$\sigma$','(3.0$\sigma$, 3.5$\sigma$)','(2.5$\sigma$, 3.0$\sigma$)'])

plt.subplots_adjust(bottom= 0.25, wspace=0.2)

plt.savefig('../figure/figure_drought_rain_affected_area_by_state.pdf')

