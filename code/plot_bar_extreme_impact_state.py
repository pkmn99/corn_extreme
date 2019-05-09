import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_figure3 import column_weighted
from plot_figure1 import weighted_mean
from load_nass_data import irrigation_percent



bin_yield = pd.read_csv('../data/result/bin_yield.csv', dtype={'FIPS':str})
bin_yield['Yield_ana_to_yield,weight'] = zip(bin_yield['Yield_ana_to_yield']*100, bin_yield['Area'])
bin_yield['Production'] = bin_yield['Yield']*bin_yield['Area']


c1 = bin_yield['Prec_sigma_bin']<4
b_drought = column_weighted(bin_yield[c1], 'State', 'Yield_ana_to_yield', 'Area')

c2 = bin_yield['Prec_sigma_bin']>12
b_rain = column_weighted(bin_yield[c2], 'State', 'Yield_ana_to_yield', 'Area')


# Load irrigation percent
irr = irrigation_percent()


# plot 
sns.set()

sns.set_style("ticks")
fig, [ax1,ax2] = plt.subplots(2,1, figsize=(8,12))

# with sns.axes_style("ticks"):
g = sns.barplot(x='State',y='Yield_ana_to_yield,weight', data=bin_yield[c1], estimator=weighted_mean,
                order=b_drought.sort_values(by='Yield_ana_to_yield_weighted')['State'].tolist(),
                orient='v', n_boot=1000, errwidth=1, ax=ax1)
g.set_xticklabels(g.get_xticklabels(), fontsize=10, rotation=90)
g.set_ylim(-50,40)

g.set_ylabel('Yield change (%)', fontsize=12)
g.set_xlabel('')
g.text(-0.1, 1, 'a', fontsize=16, transform=g.transAxes, fontweight='bold')
g.set_title('Extreme drought',fontsize=16)


g2 = sns.barplot(x='State',y='Yield_ana_to_yield,weight', data=bin_yield[c2], estimator=weighted_mean,
                order=b_rain.sort_values(by='Yield_ana_to_yield_weighted')['State'].tolist(),
                orient='v', n_boot=1000, errwidth=1, ax=ax2)
g2.set_xticklabels(g2.get_xticklabels(), fontsize=10, rotation=90)
g2.set_ylim(-50,40)

g2.set_ylabel('Yield change (%)', fontsize=12)
g2.set_xlabel('State', fontsize=12)

g2.text(-0.1, 1, 'b', fontsize=16, transform=g2.transAxes, fontweight='bold')
g2.set_title('Excessive rainfall', fontsize=16)

sns.despine()

plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.15)


plt.savefig('../figure/bar_extreme_impact_state.pdf')



#c6=bin_yield['Year']>2006
#bin_yield[c6].groupby('State').sum()['Area'].sort_values(ascending=False)[0:12].sum()
#bin_yield[c6].groupby('State').sum()['Area'].sort_values(ascending=False).sum()
#bin_yield[c6].groupby('State').sum()['Production'].sort_values(ascending=False)[0:12].sum()/ (bin_yield[c6].groupby('State').sum()['Production'].sum())
#
