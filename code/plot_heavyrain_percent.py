import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_figure1 import define_colors

# This figure shows the percentage of daily rainfall of different intensities to the growing
# season total rainfall
# Run under mygeo env

def make_plot():
    # Load and combine data
    bin_yield = pd.read_csv('/home/yanli/Project/corn_extreme/data/result/bin_yield.csv',
                            dtype={'FIPS':str})
    
    hrain = pd.read_csv('/home/yanli/Project/corn_extreme/data/result/heavy_rain_percent.csv',
                            dtype={'FIPS':str})
    
    bin_yield_hrain = bin_yield.merge(hrain,on=['FIPS','Year'])
    
    
    colors = define_colors()
    
    # Begin plot 

    fig,ax=plt.subplots(1,figsize=(8,6))
    (bin_yield_hrain.groupby('Prec_sigma_bin').mean().loc[:,slice('HPrec_percent_0',
                                                                 'HPrec_percent_8')]*100).plot(legend=False,ax=ax,
                                                                                          marker='.')
    
    legend_txt = ['<0$\sigma$', '(0$\sigma$,0.5$\sigma$)', '(0.5$\sigma$,1$\sigma$)', 
                         '(1$\sigma$,1.5$\sigma$)', '(1.5$\sigma$,2$\sigma$)', '(2$\sigma$,2.5$\sigma$)',
                         '(2.5$\sigma$,3$\sigma$)','(3$\sigma$,3.5$\sigma$)', '>3.5$\sigma$ (Heavyrain)']
    
    ax.legend(ax.lines, (legend_txt), 
              loc='upper center',frameon=False,title='Daily rainfall intensity',ncol=2)
    
    ax.set_xlabel('Precipitation anomaly of growning season', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    
    ax.set_title('Contribution of daily rainfall of different intensities to \n growing season total rainfall')
    
    x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
    x_txt.insert(0,'')
    x_txt.append('')
    
    ax.set_xlim([1.5,15.5])
    ax.set(xticks=np.arange(1.5,15.5,1), xticklabels=(x_txt))
    
    
    ax.text(0.0, -0.15, 'Extreme dry', transform=ax.transAxes, fontsize=10,
                   color=colors[0])
    ax.text(0.25, -0.15, 'Moderate dry', transform=ax.transAxes, fontsize=10,
                   color=colors[2])
    ax.text(0.5, -0.15, 'Moderate wet', transform=ax.transAxes, fontsize=10,
                   color=colors[6])
    ax.text(0.8, -0.15, 'Extreme wet', transform=ax.transAxes, fontsize=10,
                   color=colors[-1])
    
    plt.subplots_adjust(bottom=0.15,top=0.9)
    plt.savefig('../figure/figure_heavyrainf_percent.pdf')
    print('Heavy rain precent figure saved')

if __name__=='__main__':
    make_plot()

