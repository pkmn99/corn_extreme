import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from load_prism_data import load_gs_climate
from plot_figure1 import define_colors
from plot_figure3 import plot_scatter_sns, figure_data


colors = define_colors()
level = 'State'
rain_state_w, drought_state_w = figure_data(level=level)

# Load tile drainage data
tile = pd.read_csv('../../data/US_tile/drainage_shapefile.csv',
                   usecols=['STATE_NAME','FIPS','tileDra'],dtype={'FIPS':str})
tile_st = tile.groupby('STATE_NAME').mean().sort_values(by='tileDra',ascending=False).reset_index().copy()

tile_st.rename(columns={'STATE_NAME':'State'},inplace=True)
tile_st['State']=tile_st['State'].str.upper()

tile_st.loc[:,'tileDra']=tile_st.loc[:,'tileDra'] * 100

data_m = drought_state_w.merge(tile_st,on='State')

#color_var={'Prec_mean_weighted':colors[-1], 'Tmax_mean_weighted':colors[0], 
#           'Area':'#525252', soilvar+'_weighted':'#995D12'}

soilvar_label = {'ksat':'Soil saturated hydraulic conductivity (cm/hr)',
                 'clay': 'Soil clay percentage (%)',
                 'awc':'Soil available water content (m$^3$/m$^3$)'}

var_y = 'Yield_ana_to_yield_weighted'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))

plot_scatter_sns(data_m, 'tileDra', var_y,'k',
                         ax1,show_dot=True)

plot_scatter_sns(data_m,'ksat_weighted', 'tileDra','k',
                         ax2,show_dot=True)

ax1.set_ylabel('Yield change (%)', fontsize=12)
ax2.set_ylabel('Tile drainage percentage (%)', fontsize=12)

# ax1.set_title('Excessive rainfall', fontsize=14)
# ax2.set_title('Extreme drought', fontsize=14)

ax1.set_xlabel('Tile drainage percentage (%)', fontsize=12)
ax2.set_xlabel(soilvar_label['ksat'], fontsize=12)

#ax1.set_ylim(-50,30)
#ax2.set_ylim(-50,30)

plt.subplots_adjust(wspace=0.3, bottom=0.15)


plt.savefig('../figure/figure_test_%s.pdf'%level)
