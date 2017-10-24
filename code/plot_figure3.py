import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from load_prism_data import load_gs_climate
from plot_figure1 import define_colors

"""
Weighted average, see http://pbpython.com/weighted-average.html
"""
def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

"""
Get the weighted column
"""
def column_weighted(df, group, col, w):
    if isinstance(col, list):
        dfList = [df.groupby(group).apply(wavg,i,w).to_frame(i + '_weighted').reset_index() for i in col]
    #https://stackoverflow.com/questions/38089010/merge-a-list-of-pandas-dataframes        
        return reduce(lambda x, y: pd.merge(x, y, on = group), dfList)
    else:    
        return df.groupby(group).apply(wavg,col,w).to_frame(col + '_weighted').reset_index()


"""
Linear fitting, get prediction and p value
"""
def my_fitting(df, x_txt, y_txt, predict=True):
    X = df[x_txt].values
    X = sm.add_constant(X)
    y = df[y_txt].values
    mod_fit = sm.OLS(y, X, missing='drop').fit()
    if predict:
        return mod_fit.predict(), mod_fit.pvalues[1]
    else:
        return mod_fit, mod_fit.pvalues[1]


"""
Plot scatter using seaborn
"""
def plot_scatter_sns(df, x_txt, y_txt, mycolor, ax, show_dot=False):
    
    if show_dot:
        scatter_kws={"s": 20}
    else:
        scatter_kws={"s": 50, "alpha": 0}
    
    with sns.axes_style("ticks"):
#     sns.set_style("ticks")
        sns.regplot(x=x_txt, y=y_txt, data=df, ax=ax, ci=95, color=mycolor, 
                    scatter_kws=scatter_kws)
        sns.despine()    

    y_p, p = my_fitting(df, x_txt, y_txt, predict=False)
    r = df.corr()[y_txt][x_txt]
    
    # if p < 0.05, add * 
    if p < 0.05:
        r_txt = r'$r$=%.2f$^%s$'%(np.round(r,2),'*')
    else:
        r_txt = r'$r$=%.2f'%(np.round(r,3))    

    ax.text(0.7, 0.9, r_txt, transform=ax.transAxes)
    ax.set_ylabel('')
    ax.set_xlabel('')
    if level !='FIPS':
        ax.set_ylim(-50,30)
    else:
        ax.set_ylim(-100,100)
    ax.axes.tick_params(axis='both',labelsize=10)


"""
Prepare figure data to plot, level can be 'State' or 'FIPS'
level = 'FIPS'
"""
def figure_data(level='State'):
    # Load growing season mean climate
    tmax_gs = load_gs_climate(var='tmax', rerun=False)
    prec_gs = load_gs_climate(var='ppt', rerun=False)
    
    tmax_prec_mean = pd.concat([tmax_gs.mean(),prec_gs.mean()],
                               axis=1).reset_index().rename(columns={'index':'FIPS',
                                                                     0:'Tmax_mean',
                                                                     1:'Prec_mean'})

    bin_yield = pd.read_csv('../data/result/bin_yield.csv', dtype={'FIPS':str})
    
    soil = pd.read_csv('/home/yanli/Project/data/GEE/US_soil_corn_county/US_county_corn_soil_property.csv',
                       dtype={'FIPS':str})
    
    bin_yield = bin_yield.merge(soil, on='FIPS').merge(tmax_prec_mean, on='FIPS')
    
    
    # Weighted soil properties up to 30cm depth
    bin_yield['ksat'] = (bin_yield['ksat_mean_0_5']/6 + bin_yield['ksat_mean_5_15']/3 
                         + bin_yield['ksat_mean_15_30']*0.5)
        
    bin_yield['clay'] = (bin_yield['clay_mean_0_5']/6 + bin_yield['clay_mean_5_15']/3 
                         + bin_yield['clay_mean_15_30']*0.5)
    
    bin_yield['awc'] = (bin_yield['awc_mean_0_5']/6 + bin_yield['awc_mean_5_15']/3 + 
                        bin_yield['awc_mean_15_30']*0.5)

    var_list = ['Prec','Tmax','ksat','clay','awc','Prec_mean','Tmax_mean']

#                'clay_mean_0_5','ksat_mean_0_5','awc_mean_0_5',
#                'clay_mean_5_15','ksat_mean_5_15','awc_mean_5_15',
#                'clay_mean_15_30','ksat_mean_15_30','awc_mean_15_30',
#                'clay_mean_30_60','ksat_mean_30_60','awc_mean_30_60',
#                'clay_mean_60_100','ksat_mean_60_100','awc_mean_60_100']
    
    c1 = bin_yield['Prec_sigma_bin']>12
    
    rain_state_w = (column_weighted(bin_yield[c1], level, 'Yield_ana_to_yield','Area')).\
                    merge(column_weighted(bin_yield, level, var_list, 'Area')).\
                    merge(bin_yield.groupby(level).mean()['Area'].reset_index())
    
    c2 = bin_yield['Prec_sigma_bin']<4
    
    drought_state_w = (column_weighted(bin_yield[c2], level, 'Yield_ana_to_yield','Area')).\
                        merge(column_weighted(bin_yield, level, var_list, 'Area')).\
                        merge(bin_yield.groupby(level).mean()['Area'].reset_index())
            
    rain_state_w['Yield_ana_to_yield_weighted'] = rain_state_w['Yield_ana_to_yield_weighted']*100     
    drought_state_w['Yield_ana_to_yield_weighted'] = drought_state_w['Yield_ana_to_yield_weighted']*100     
    
   # if level!='FIPS':
    rain_state_w['Area'] = rain_state_w['Area']/1000     
    drought_state_w['Area'] = drought_state_w['Area']/1000     

    return rain_state_w, drought_state_w


def make_plot():
    # colors for prec, tmax, area, soil
    color_var={'Prec_mean_weighted':colors[-1], 'Tmax_mean_weighted':colors[0], 
               'Area':'#525252', 'awc_weighted':'#995D12'}

    x_txt = ['Prec_mean_weighted','Tmax_mean_weighted','Area','awc_weighted']
    
    fig, axes = plt.subplots(2,len(x_txt), figsize=(14,7))
    
    var_y = 'Yield_ana_to_yield_weighted'

    show_name = False
    show_dot = True
   
    # plot rain, first row 
    for i in range(len(x_txt)):
        plot_scatter_sns(rain_state_w, x_txt[i], var_y, color_var[x_txt[i]],
                         axes.flatten()[i],show_dot=show_dot)

    # plot drought, second row 
    for i in range(len(x_txt),len(x_txt)*2):
        plot_scatter_sns(drought_state_w, x_txt[i-len(x_txt)], var_y, color_var[x_txt[i-len(x_txt)]], 
        axes.flatten()[i],show_dot=show_dot)
    
    # Control limit        
    if level != 'FIPS':
        axes.flatten()[0].set_xlim(0, 700) # Prec
        axes.flatten()[4].set_xlim(0, 700)
        
        axes.flatten()[2].set_xlim(-10, 140) # Area
        axes.flatten()[6].set_xlim(-10, 140)
    
    axes.flatten()[0].set_ylabel('Yield change (%)', fontsize=12)
    axes.flatten()[4].set_ylabel('Yield change (%)', fontsize=12)
    
    
    # Add xlabel
    xlabel_txt = [u'Precipitation (mm)', 'Maximum temperature (${^\circ}$C)', 'Harvest area (10$^3$acres)', 
                  'Soil available water content (m$^3$/m$^3$)']
    for i in range(len(x_txt)):
        axes.flatten()[i].set_xlabel(xlabel_txt[i], fontsize=12)
        axes.flatten()[i+4].set_xlabel(xlabel_txt[i], fontsize=12)
        
    # Add panel char        
    panel_txt = [chr(i) for i in range(ord('a'), ord('h')+1)]  
    for i in range(len(x_txt)*2):
        axes.flatten()[i].text(-0.2, 1, panel_txt[i], fontsize=14, transform=axes.flatten()[i].transAxes,
                               fontweight='bold')
    
    plt.text(0.5,0.95, 'Extreme rainfall', fontsize=16, transform=fig.transFigure, 
             ha='center')        
    plt.text(0.5,0.475, 'Extreme drought', fontsize=16, transform=fig.transFigure, 
             ha='center')        
    
    plt.subplots_adjust(left=0.075, right=0.95, top=0.925, hspace=0.4, wspace=0.3)
    
    plt.savefig('../figure/figure3_%s.pdf'%level)

    print('figure file %s level saved'%level)


if __name__ == "__main__":
    colors = define_colors()
    level = 'FIPS'
   # level = 'State'
    rain_state_w, drought_state_w = figure_data(level=level)
    make_plot()

