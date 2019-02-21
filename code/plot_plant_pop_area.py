import matplotlib.pyplot as plt
import pandas as pd
from load_nass_data import load_nass_county_data
from plot_figure3 import plot_scatter_sns


# Load data
corn_pop = load_nass_county_data('corn', 'grain_population', 'allstates', 1981, 2016)
corn_pop.rename(columns={'Value':'Pop'}, inplace=True)

corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1981, 2016)
corn_area.rename(columns={'Value':'Area'}, inplace=True)
corn_area = corn_area.dropna()

# Combine
corn_pop_area = corn_pop.merge(corn_area.groupby(['Year','State']).mean().reset_index(), on=['Year','State'])
corn_pop_area['Area'] = corn_pop_area['Area']/1000

fig, [ax1,ax2] = plt.subplots(1,2,figsize=(8,4))

c3 = corn_pop_area['Year'] <= 2000
plot_scatter_sns(corn_pop_area[c3], 'Area', 'Pop', 'k', ax1, show_dot=True)
ax1.set_ylim(20000,35000)
ax1.set_ylabel('Plant population (plants/acre)')
ax1.set_xlabel('Harvest area (10$^3$acres)')
ax1.set_title('1981-2000')
ax1.text(-0.15, 1, 'a', fontsize=16, transform=ax1.transAxes, fontweight='bold')

c4 = corn_pop_area['Year'] > 2000
plot_scatter_sns(corn_pop_area[c4], 'Area', 'Pop', 'k', ax2, show_dot=True)
ax2.set_ylim(20000,35000)
#ax2.set_ylabel('Plant population (plants/acre)')
ax2.set_xlabel('Harvest area (10$^3$acres)')
ax2.set_title('2001-2016')
ax2.text(-0.15, 1, 'b', fontsize=16, transform=ax2.transAxes, fontweight='bold')

plt.subplots_adjust(wspace=0.3, bottom=0.15)

plt.savefig('../figure/figure_plant_pop_area.pdf')

