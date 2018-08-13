import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Plot the correlation between pre-growing season soil moisture and the yield change
# Run with the mygeo or env_test to use the new style

"""
Load monthly data and then convert the data to column
# Modified based on the prism data 
"""
def convert_to_gs_monthly(df_mon,var_name,month_start=1,month_end=8):
    # Select growing season
    df_gs = df_mon[(df_mon.index.month>=month_start)&(df_mon.index.month<=month_end)]

    # make some rearrangement of the data layout 
    df_gs_1 = df_gs.stack().to_frame('value').reset_index()
    df_gs_1.rename(columns={'level_1':'FIPS'}, inplace=True)

    # Add year and month as column
    df_gs_1['year'] = df_gs_1['level_0'].apply(lambda x: x.year)
    df_gs_1['mon'] = df_gs_1['level_0'].apply(lambda x: x.month)

    # Seperate monthly lst as column by mutle-index and pivot
    df_gs_2 = df_gs_1.iloc[:,1::].set_index(['year','FIPS']).pivot(columns='mon')

    # drop multi-index of columns
    df_gs_2.columns = df_gs_2.columns.droplevel(0)

        # rename lst column
    df_gs_2 = df_gs_2.reset_index().rename(columns={1:'%s1'%var_name,
                                                    2:'%s2'%var_name,
                                                    3:'%s3'%var_name,
                                                    4:'%s4'%var_name,
                                                    5:'%s5'%var_name,
                                                    6:'%s6'%var_name,
                                                    7:'%s7'%var_name,
                                                    8:'%s8'%var_name})
    return df_gs_2


def get_corr(df, x_txt, y_txt):
    return pearsonr(df.loc[:,[x_txt,y_txt]].dropna().iloc[:,0],
                                df.loc[:,[x_txt,y_txt]].dropna().iloc[:,1])
# Prepare data to plot the figure
def figure_data():
    d_test = pd.read_csv('../data/SOILMOISTURE_monthly_1981-2016_county.csv',index_col=0,parse_dates=True)
    dm = convert_to_gs_monthly(d_test,'SM')
    dm.rename(columns={'year':'Year'},inplace=True)
    
    bin_yield = pd.read_csv('../data/result/bin_yield.csv', dtype={'FIPS':str})
    d = bin_yield.merge(dm,on=['Year','FIPS'])
    return d

# Determine text location based on the sign
def get_va(v):
    if v>0:
        va_txt = 'bottom'
    else:
        va_txt = 'top'
        
    return va_txt

def make_plot():
    d = figure_data()
    irr_states = ['COLORADO', 'DELAWARE', 'KANSAS', 'MONTANA', 'NEBRASKA', 'TEXAS',
                   'SOUTH DAKOTA', 'NEW MEXICO', 'NORTH DAKOTA', 'OKLAHOMA', 'WYOMING']

    # Excessvie rainfall data
    c=d['Prec_sigma_bin']>12

    # Get state where excessive rainfall has negative yield impact
    sname_negative = d[c].groupby('State').mean()['Yield_ana_to_yield'].sort_values()[0:20].index.values
    c3=d['State'].isin(sname_negative)
    
    d_corr_rain = d.loc[c&c3,['SM1','SM2','SM3','SM4','SM5','SM6','SM7','SM8',
                              'Yield_ana_to_yield']].corr()
    
    p_rain = [get_corr(d.loc[c&c3], s, 'Yield_ana_to_yield')[1] for s in ['SM1','SM2','SM3','SM4']]
    r_rain = [get_corr(d.loc[c&c3], s, 'Yield_ana_to_yield')[0] for s in ['SM1','SM2','SM3','SM4'] ]


    # Extreme drought data
    c=d['Prec_sigma_bin']<4
    c3=~d['State'].isin(irr_states) # Not irrigation states
    
    d_corr_dry = d.loc[c&c3,['SM1','SM2','SM3','SM4','SM5','SM6','SM7','SM8',
                             'Yield_ana_to_yield']].corr()
    
    p_dry = [get_corr(d.loc[c&c3], s, 'Yield_ana_to_yield')[1] for s in ['SM1','SM2','SM3','SM4']]
    r_dry = [get_corr(d.loc[c&c3], s, 'Yield_ana_to_yield')[0] for s in ['SM1','SM2','SM3','SM4']]
    
    # Begin plot
    fig, [ax1,ax2] = plt.subplots(1,2,sharey=True,figsize=(8,4))
    d_corr_dry.loc['Yield_ana_to_yield',slice('SM1','SM4')].plot.bar(ax=ax1)
    d_corr_rain.loc['Yield_ana_to_yield',slice('SM1','SM4')].plot.bar(ax=ax2)
    
    ax1.set_title('Extreme drought')
    ax2.set_title('Excessive rainfall')
    
   # ax1.set_ylabel('$r$')
    ax1.set_ylabel('$r$ between yield change and soil moisture')

    
    ax1.set_xticklabels(['Janurary','Febuary','March','April'],rotation=0)
    ax2.set_xticklabels(['Janurary','Febuary','March','April'],rotation=0)
    
    for i in range(0,4):
        if p_dry[i] < 0.05:
            ax1.text(i,r_dry[i],'*',ha='center',va=get_va(r_dry[i]))
        if p_rain[i] < 0.05:
            ax2.text(i,r_rain[i],'*',ha='center',va=get_va(r_rain[i]))
            
    ax1.text(-0.1, 1.03, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.1, 1.03, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    
    plt.subplots_adjust(top=0.85)

    plt.savefig('../figure/figure_preseason_soilmoisture.pdf')
    print('Figure saved')

if __name__=='__main__':
    make_plot()
