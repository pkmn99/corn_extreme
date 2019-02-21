import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from load_prism_data import load_gs_climate
from plot_figure1 import define_colors
from plot_figure3 import plot_scatter_sns, figure_data


colors = define_colors()
level = 'FIPS'
rain_state_w, drought_state_w = figure_data(level=level)

#color_var={'Prec_mean_weighted':colors[-1], 'Tmax_mean_weighted':colors[0], 
#           'Area':'#525252', soilvar+'_weighted':'#995D12'}

soilvar_label = {'ksat':'Soil saturated hydraulic conductivity (cm/hr)',
                 'clay': 'Soil clay percentage (%)',
                 'awc':'Soil available water content (m$^3$/m$^3$)'}

var_y = 'Yield_ana_to_yield_weighted'
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5,8))

plot_scatter_sns(rain_state_w,'clay_weighted', var_y,'#995D12',
                         ax1,show_dot=True)

plot_scatter_sns(drought_state_w,'clay_weighted', var_y,'#995D12',
                         ax2,show_dot=True)

ax1.set_ylabel('Yield change (%)', fontsize=12)
ax2.set_ylabel('Yield change (%)', fontsize=12)

ax1.set_title('Excessive rainfall', fontsize=14)
ax2.set_title('Extreme drought', fontsize=14)

ax1.set_xlabel('Soil clay percentage (%)', fontsize=12)
ax2.set_xlabel('Soil clay percentage (%)', fontsize=12)

#ax1.set_ylim(-50,30)
#ax2.set_ylim(-50,30)

plt.subplots_adjust(left=0.15, hspace=0.3)

plt.savefig('../figure/figure_clay_impact_%s.pdf'%level)
