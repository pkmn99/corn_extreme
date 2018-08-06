import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_figure1 import define_colors
from plot_figure3 import column_weighted


# Append a repeatative value for plt.step function
def append_value(ds):
    return ds.append(pd.Series(ds.loc[15],index=[16]))

# Calculate bin model values
def fig_data(climate_data='agmerra'):
    if climate_data=='agmerra':
       data_tag = ''
    else:
       data_tag = '_'+climate_data 

    agmip = pd.read_csv('../data/result/agmip_obs_yield_full%s.csv'%data_tag)
    agmip.iloc[:,7::] = agmip.iloc[:,7::]*100
    agmip.iloc[:,3] = agmip.iloc[:,3]*100

    model_names = ['obs','cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
                      'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']

    b_prec = pd.DataFrame(np.zeros([14,13]), index=range(2,16), columns=model_names)
    b_tmax = pd.DataFrame(np.zeros([14,13]), index=range(2,16), columns=model_names)

    for m in model_names:
        temp=agmip[['Prec_sigma_bin','Tmax_sigma_bin',m,'Area']].dropna()

        b1 = column_weighted(temp, 'Prec_sigma_bin', m, 'Area')
        b2 = column_weighted(temp, 'Tmax_sigma_bin', m, 'Area')

        b_prec[m] = b1.set_index('Prec_sigma_bin') 
        b_tmax[m] = b2.set_index('Tmax_sigma_bin')
    return b_prec, b_tmax

def make_plot(climate_data='agmerra'):

    sns.set()
    sns.set_context("notebook")
    sns.set_style('ticks')    
    
    colors = define_colors()
    b_prec, b_tmax = fig_data(climate_data=climate_data)
    #b_prec = pd.read_csv('../data/result/agmip_prec_bin.csv', index_col=0)
    
    
    # Sample plot use step, show response of individual model but no names
    palette = sns.color_palette()
    
    x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
    x_txt.insert(0,'')
    x_txt.append('')
    
    
    fig, ax = plt.subplots(1)
    
    
    ax.plot([1,15],[0,0], color='k', lw=0.5, linestyle='--')    
    
    
    temp = append_value(b_prec['obs'])
    l1, = ax.step(range(1,16), temp, where='post', color='#1A1834', lw=3, alpha=0.9)
    
    temp = append_value(b_prec.iloc[:,1::].median(axis=1))
    
    l2, = ax.step(range(1,16), temp, where='post', color=palette[1], lw=3)
    
    # for i in range(1,13):
    #     b_prec.iloc[:,i]
    #     l3, = ax.step(range(1,16)+np.random.rand(15)/10, append_value(b_prec.iloc[:,i]), where='post', color=palette[1], 
    #                   lw=0.75, alpha=0.5,linestyle='-')
    
        
    # ax.set_ylim(-0.5,0.5)
    ax.set_ylim(-75,75)
    
    ax.set_xlim(1,15)
    
    ax.set(xticks=np.arange(1,15,1), xticklabels=(x_txt))
    
    ax.set_ylabel('Yield change (%)', fontsize=12)
    ax.set_xlabel('Precipitation anomaly ($\sigma$)',labelpad=15, fontsize=12)
    
    ax.text(0.00+0.04, -0.1, 'Extremely dry', transform=ax.transAxes, fontsize=10,
                   color=colors[0])
    ax.text(0.25+0.04, -0.1, 'Moderate dry', transform=ax.transAxes, fontsize=10,
                   color=colors[2])
    ax.text(0.5+0.04, -0.1, 'Moderate wet', transform=ax.transAxes, fontsize=10,
                   color=colors[6])
    ax.text(0.75+0.04, -0.1, 'Extremely wet', transform=ax.transAxes, fontsize=10,
                   color=colors[-1])
    
    [t.set_color(colors[0]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i<3]
    [t.set_color(colors[-1]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i>=11]
    
    
    # max and min range
    p1 = ax.fill_between(range(1,16), append_value(b_prec.iloc[:,1::].min(axis=1)), 
                     append_value(b_prec.iloc[:,1::].max(axis=1)), alpha=0.3, color=palette[1],
                   step='post')
    
    # interquantile range
    p2= ax.fill_between(range(1,16), append_value(b_prec.iloc[:,1::].quantile(0.25,axis=1)), 
                     append_value(b_prec.iloc[:,1::].quantile(0.75,axis=1)), alpha=0.5, color=palette[1],
                   step='post')
    
    ax.legend([l1,l2,p2,p1], ['Obs','AgMIP median','AgMIP interquantile','AgMIP range'], loc='upper left')
    
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig('../figure/figure4_%s.pdf'%climate_data)
    print('Figure saved for %s'%climate_data)


if __name__ == "__main__":
    make_plot(climate_data='wfdei.gpcc')
