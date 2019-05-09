import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn.apionly as sns
import seaborn as sns
import scikits.bootstrap as bootstrap  


def weighted_mean(x, **kws):
    val, weight = map(np.asarray, zip(*x))
    return (val * weight).sum() / weight.sum()

def define_colors():
    # Define color
    color1 = sns.diverging_palette(10, 257, l=50, s=85, n=2)
    color2 = sns.diverging_palette(10, 257, l=70, s=85, n=9)[2]
    color3 = sns.diverging_palette(10, 257, l=70, s=85, n=9)[6]
    colors = [color1[0]]*3 + [color2]*5 + [color3]*4 +[color1[-1]]*2
    return colors

def make_plot():

    bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})
    bin_yield['Yield_ana_to_yield'] = bin_yield['Yield_ana_to_yield'] * 100 # convert to %
    bin_yield['Yield_ana_to_yield_area'] = bin_yield['Yield_ana_to_yield'] * bin_yield['Area']
    bin_yield['Yield_ana_to_yield,weight'] = zip(bin_yield['Yield_ana_to_yield'], bin_yield['Area'])

    fig, axes = plt.subplots(1, figsize=(8,6))
    
    x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
    x_txt.insert(0,'')
    x_txt.append('')
    
    ##################################### Panel A

   # sns.barplot(x='Prec_sigma_bin', y='Yield_ana_to_yield', data=bin_yield, 
   #             palette=colors, ci=95, orient='v', saturation=1, 
   #             ax=axes[0,0])

    sns.barplot(x='Tmax_sigma_bin', y='Yield_ana_to_yield,weight', estimator=weighted_mean, 
                data=bin_yield, palette=colors[::-1], ci=95, orient='v', saturation=1, 
                ax=axes)

    sns.despine()
    axes.set(xticks=np.arange(-0.5,14.5,1), xticklabels=(x_txt))
    
    axes.set_ylabel('Yield change (%)', fontsize=12)
    axes.set_xlabel('Temperature anomaly ($\sigma$)',labelpad=17, fontsize=12)
    axes.text(0.0, -0.1, 'Extreme cold', transform=axes.transAxes, fontsize=10,
                   color=colors[-1])
    axes.text(0.25, -0.1, 'Moderate cold', transform=axes.transAxes, fontsize=10,
                   color=colors[-4])
    axes.text(0.5, -0.1, 'Moderate hot', transform=axes.transAxes, fontsize=10,
                   color=colors[4])
    axes.text(0.8, -0.1, 'Extreme hot', transform=axes.transAxes, fontsize=10,
                   color=colors[0])

#    axes[0,0].text(-0.15, 1, 'a', fontsize=16, transform=axes[0,0].transAxes, fontweight='bold')
     
    # Calculte exteme drought and rainfall impact to show on bar chart
#    c5 = b_drought_rain['Type'] == 'Dry'
#    v_drought = (b_drought_rain[c5]['Yield_ana_to_yield'] * b_drought_rain[c5]['Area']).sum()/b_drought_rain[c5]['Area'].sum()
#    
#    c6 = b_drought_rain['Type'] == 'Rain'
#    v_rain = (b_drought_rain[c6]['Yield_ana_to_yield'] * b_drought_rain[c6]['Area']).sum()/b_drought_rain[c6]['Area'].sum()
#
#
#    axes.text(0.02, 0.9, "{0:.1f}%".format(v_drought), transform=axes[0,0].transAxes,
#                   fontsize=10, color=colors[0])
#
#    axes.text(0.85, 0.9,"{0:.1f}%".format(v_rain), transform=axes[0,0].transAxes,
#                   fontsize=10, color=colors[-1])
#
    
    # axes[0,0].xaxis.set_ticks_position('bottom')
    
    
    # Change xlabel color
    [t.set_color(colors[-1]) for i,t in enumerate(axes.xaxis.get_ticklabels()) if i<3]
    [t.set_color(colors[0]) for i,t in enumerate(axes.xaxis.get_ticklabels()) if i>=11]
    # [t.set_color(colors[-1]) for t in axes[0,0].xaxis.get_ticklabels()[-4::]]
    
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.3)
    
    plt.savefig('../figure/figure_bar_tmax.pdf')

if __name__ == "__main__":
    colors = define_colors()
    sns.set_context("notebook")
    sns.set_style("ticks")

    make_plot()
