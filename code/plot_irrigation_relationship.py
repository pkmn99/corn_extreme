import matplotlib.pyplot as plt
from load_nass_data import irrigation_percent
from plot_figure3 import figure_data
from plot_figure3 import plot_scatter_sns

irr = irrigation_percent()
rain_state_w, drought_state_w = figure_data()

df_irr = drought_state_w.merge(irr.rename('irr').to_frame().reset_index())

fig, ax1 = plt.subplots(1,1)
plot_scatter_sns(df_irr, 'irr', 'Yield_ana_to_yield_weighted', 'k', ax1, show_dot=True)
ax1.set_ylabel('Yield change (%)',fontsize=14)
ax1.set_xlabel('Irrigation fraction',fontsize=14)
# ax1.text(-0.15, 1, 'a', fontsize=16, transform=ax1.transAxes, fontweight='bold')
plt.savefig('../figure/figure_irrigation_relationship.pdf')
print('figure saved')
