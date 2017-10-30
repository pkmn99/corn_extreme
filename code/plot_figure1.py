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
    colors = [color1[0]]*2 + [color2]*4 + [color3]*5 +[color1[-1]]*3
    return colors


def fig_data():
    # Panel A
    bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})
    bin_yield['Yield_ana_to_yield'] = bin_yield['Yield_ana_to_yield'] * 100 # convert to %
    bin_yield['Yield_ana_to_yield_area'] = bin_yield['Yield_ana_to_yield'] * bin_yield['Area']
    bin_yield['Yield_ana_to_yield,weight'] = zip(bin_yield['Yield_ana_to_yield'], bin_yield['Area'])


    # rma loss 
    bin_yield_rma = pd.read_csv('../data/result/bin_yield_rma.csv',index_col=0)
    
    bin_yield_rma[['loss_ratio_Heat','loss_ratio_Drought','loss_ratio_Excess Moisture/Precip/Rain',
                   'loss_ratio_Cold Wet Weather']] = \
        bin_yield_rma[['loss_ratio_Heat','loss_ratio_Drought',
                       'loss_ratio_Excess Moisture/Precip/Rain',
                       'loss_ratio_Cold Wet Weather']].multiply(bin_yield_rma['Area'],axis=0)
    # Panel B: loss ratio
    loss_ratio_Prec = (bin_yield_rma.groupby('Prec_sigma_bin').sum()[['loss_ratio_Drought', \
                                                'loss_ratio_Heat',\
                                                'loss_ratio_Excess Moisture/Precip/Rain',\
                                                'loss_ratio_Cold Wet Weather'
                                                  ]]).divide(bin_yield_rma.groupby('Prec_sigma_bin').sum()['Area'],axis=0)

    loss_ratio_Prec.columns=['Drought', 'Heat', 'Excess Rain','Cold Weather']
    
    # Panel C: temperature interaction
    c3 = bin_yield['Prec_sigma_bin']<4 # all drought
    c4 = (bin_yield['Tmax_sigma_bin']>12)&(bin_yield['Prec_sigma_bin']<4) # hot and dry
    c5 = (bin_yield['Tmax_sigma_bin']<=12)&(bin_yield['Prec_sigma_bin']<4) # dry but not hot
    
    b_dry = bin_yield[c3].copy()
    b_dry['Type'] = 'Dry'
    
    b_dry_hot = bin_yield[c4].copy()
    b_dry_hot['Type'] = 'Dry+Hot'
    
    b_dry_nohot = bin_yield[c5].copy()
    b_dry_nohot['Type'] = 'Dry-Hot'
    
    b_drought = pd.concat([b_dry, b_dry_hot, b_dry_nohot])
    b_drought['Event'] = 'Drought'
    
    c3 = bin_yield['Prec_sigma_bin']>12 # all heavy rainfall
    c4 = (bin_yield['Tmax_sigma_bin']>=4)&(bin_yield['Prec_sigma_bin']>12) # rain not cold
    c5 = (bin_yield['Tmax_sigma_bin']<4)&(bin_yield['Prec_sigma_bin']>12) # rain and cold
    
    b_wet = bin_yield[c3].copy()
    b_wet['Type'] = 'Rain'
    
    b_wet_nocold = bin_yield[c4].copy()
    b_wet_nocold['Type'] = 'Rain-cold'
    
    b_wet_cold = bin_yield[c5].copy()
    b_wet_cold['Type'] = 'Rain+cold'
    
    b_rain = pd.concat([b_wet, b_wet_nocold, b_wet_cold])
    b_rain['Event'] = 'Extreme rain'
    b_drought_rain = pd.concat([b_drought[['Event','Type','Yield_ana_to_yield','Area']], 
                                b_rain[['Event','Type', 'Yield_ana_to_yield','Area']]])
    b_drought_rain['Yield_ana_to_yield,weight'] = zip(b_drought_rain['Yield_ana_to_yield'], b_drought_rain['Area'])
 
#    drought_rain_table = pd.DataFrame(np.zeros([6,4]), index=['Dry','Dry+hot','Dry-hot','Rain','Rain+cold',
#                                      'Rain-cold'],columns=['mean','ci_low','ci_high','ci'])
#    # Weighted mean
#    drought_rain_table.iloc[0,0] = np.average(b_dry['Yield_ana_to_yield'],weights=b_dry['Area'])
#    drought_rain_table.iloc[1,0] = np.average(b_dry_hot['Yield_ana_to_yield'],weights=b_dry_hot['Area'])
#    drought_rain_table.iloc[2,0] = np.average(b_dry_nohot['Yield_ana_to_yield'],weights=b_dry_nohot['Area'])
#    drought_rain_table.iloc[3,0] = np.average(b_wet['Yield_ana_to_yield'],weights=b_wet['Area'])
#    drought_rain_table.iloc[4,0] = np.average(b_wet_cold['Yield_ana_to_yield'],weights=b_wet_cold['Area'])
#    drought_rain_table.iloc[5,0] = np.average(b_wet_nocold['Yield_ana_to_yield'],weights=b_wet_nocold['Area'])
#    
#    # bootstramp CI
#    drought_rain_table.iloc[0,1:3] = bootstrap.ci(data=b_dry['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    drought_rain_table.iloc[1,1:3] = bootstrap.ci(data=b_dry_hot['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    drought_rain_table.iloc[2,1:3] = bootstrap.ci(data=b_dry_nohot['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    drought_rain_table.iloc[3,1:3] = bootstrap.ci(data=b_wet['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    drought_rain_table.iloc[4,1:3] = bootstrap.ci(data=b_wet_cold['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    drought_rain_table.iloc[5,1:3] = bootstrap.ci(data=b_wet_nocold['Yield_ana_to_yield,weight'], statfunction=weighted_mean, n_samples=1000,method='pi') 
#    
#    # Ci value 
#    drought_rain_table['ci'] = drought_rain_table['mean'] -  drought_rain_table['ci_low']    

    # Panel 4 data preparation: loss ratio by month
    from load_rma_data import load_rma_loss_ratio_cause_month
    
    loss_ratio_cause_month = load_rma_loss_ratio_cause_month(rerun=False)
    loss_ratio_cause_month.rename(columns={'Commodity Year':'Year'},inplace=True)
    
    loss_ratio_cause_month_bin = loss_ratio_cause_month.merge(bin_yield, on=['FIPS','Year'])
    
    
    c_gs_month = loss_ratio_cause_month_bin['Month of Loss Abbreviation'].isin(['MAY','JUN','JUL','AUG'])
    c1 = loss_ratio_cause_month_bin['Prec_sigma_bin']>12
    c2 = loss_ratio_cause_month_bin['Prec_sigma_bin']<4
    c3 = loss_ratio_cause_month_bin['Damage Cause Description']=='Drought'
    c4 = loss_ratio_cause_month_bin['Damage Cause Description']=='Excess Moisture/Precip/Rain'
    
    # Create a dataframe that only contains only drought, excessive rainfall loss ratio in extreme years
    loss_temp = pd.concat([loss_ratio_cause_month_bin[c_gs_month&c1&c4][['Damage Cause Description', 
                                                  'Month of Loss Abbreviation','Loss ratio by cause and month','Area']],
             loss_ratio_cause_month_bin[c_gs_month&c2&c3][['Damage Cause Description', 
                                                  'Month of Loss Abbreviation','Loss ratio by cause and month','Area']]])
    
    loss_temp['loss_ratio,weight'] = zip(loss_temp['Loss ratio by cause and month'], loss_temp['Area'])

    return bin_yield, loss_ratio_Prec, b_drought_rain, loss_temp

def plot_panel_c(b_drought_rain, axes):

    f1 = sns.barplot(x='Type', y='Yield_ana_to_yield,weight',
                     data=b_drought_rain, estimator=weighted_mean,
                     order=['Dry','Dry+Hot','Dry-Hot','Rain','Rain+cold','Rain-cold'],
                     ci=95, orient='v', ax=axes)
    
   # order=['dry','dry_hot','dry_nohot','wet','wet_cold','wet_nocold'],

    # Move bar
    for i, patch in enumerate(axes.patches):
            if i >= 3:
                patch.set_x(patch.get_x() + 0.5)
    
    # Change bar color
    for i, patch in enumerate(axes.patches):
            if i >= 3:
                patch.set_facecolor(colors[-1])
            else:    
                patch.set_facecolor(colors[0])
                
    # Move error bar
    for i, line in enumerate(axes.lines):
            if i >= 3:
                line.set_xdata(line.get_xdata() + 0.5)            
    
    # Rearrange xtick and label            
    x_txt = ['Dry(all)', 'Dry+hot', 'Dry-hot', 'Rain(all)','Rain+cold','Rain-cold']
    xtick = np.arange(0,6.0)
    xtick[3::] = xtick[3::] + 0.5
    
    f1.set(xticks=xtick, xticklabels=x_txt, xlim=(-0.6,6+0.1))            
    f1.set_ylabel('Yield change (%)', fontsize=12)
    f1.set_xlabel('Interactions with temperature', fontsize=12)
    
    axes.legend((axes.patches[0], axes.patches[3]), ('Drought', 'Extreme rainfall'), loc='upper left')


def make_plot():
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    bin_yield, loss_ratio_Prec, b_drought_rain, loss_temp = fig_data()
    
    x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
    x_txt.insert(0,'')
    x_txt.append('')
    
    ##################################### Panel A

   # sns.barplot(x='Prec_sigma_bin', y='Yield_ana_to_yield', data=bin_yield, 
   #             palette=colors, ci=95, orient='v', saturation=1, 
   #             ax=axes[0,0])
    sns.barplot(x='Prec_sigma_bin', y='Yield_ana_to_yield,weight', estimator=weighted_mean, 
                data=bin_yield, palette=colors, ci=95, orient='v', saturation=1, 
                ax=axes[0,0])

    sns.despine()
    axes[0,0].set(xticks=np.arange(-0.5,14.5,1), xticklabels=(x_txt))
    
    axes[0,0].set_ylabel('Yield change (%)', fontsize=12)
    axes[0,0].set_xlabel('Precipitation deviation ($\sigma$)',labelpad=15, fontsize=12)
    axes[0,0].text(0.0, -0.15, 'Extremely dry', transform=axes[0,0].transAxes, fontsize=10,
                   color=colors[0])
    axes[0,0].text(0.25, -0.15, 'Normal dry', transform=axes[0,0].transAxes, fontsize=10,
                   color=colors[2])
    axes[0,0].text(0.5, -0.15, 'Normal wet', transform=axes[0,0].transAxes, fontsize=10,
                   color=colors[6])
    axes[0,0].text(0.75, -0.15, 'Extremely wet', transform=axes[0,0].transAxes, fontsize=10,
                   color=colors[-1])

    axes[0,0].text(-0.15, 1, 'a', fontsize=16, transform=axes[0,0].transAxes, fontweight='bold')

    
    # axes[0,0].xaxis.set_ticks_position('bottom')
    
    
    # Change xlabel color
    [t.set_color(colors[0]) for i,t in enumerate(axes[0,0].xaxis.get_ticklabels()) if i<3]
    [t.set_color(colors[-1]) for i,t in enumerate(axes[0,0].xaxis.get_ticklabels()) if i>=11]
    # [t.set_color(colors[-1]) for t in axes[0,0].xaxis.get_ticklabels()[-4::]]
    
    
    # f1.set(ylim=[-0.4,0.4])
    
    # ax1.set_title('Yield change')
    
    ######################################### Panel B
    
    loss_ratio_Prec.plot.bar(width=0.75, stacked=True, ax=axes[0,1], color=['#e22c3d','#89160F','#3873DF','#8A62AE'])
    axes[0,1].set_xticks(np.arange(-0.5,14.5,1))
    axes[0,1].set_xlim(-0.5,13.5)
    axes[0,1].set_xticklabels(x_txt, rotation=0)
    axes[0,1].set_xlabel('Precipitation deviation ($\sigma$)', labelpad=15, fontsize=12)
    axes[0,1].set_ylabel('Loss ratio', fontsize=12)
    
    # Change xlabel color
    [t.set_color(colors[0]) for i,t in enumerate(axes[0,1].xaxis.get_ticklabels()) if i<3]
    [t.set_color(colors[-1]) for i,t in enumerate(axes[0,1].xaxis.get_ticklabels()) if i>=11]

    # Add tick color
    axes[0,1].text(0.0, -0.15, 'Extremely dry', transform=axes[0,1].transAxes, fontsize=10,
                   color=colors[0])
    axes[0,1].text(0.25, -0.15, 'Normal dry', transform=axes[0,1].transAxes, fontsize=10,
                   color=colors[2])
    axes[0,1].text(0.5, -0.15, 'Normal wet', transform=axes[0,1].transAxes, fontsize=10,
                   color=colors[6])
    axes[0,1].text(0.75, -0.15, 'Extremely wet', transform=axes[0,1].transAxes, fontsize=10,
                   color=colors[-1])

    axes[0,1].text(-0.15, 1, 'b', fontsize=16, transform=axes[0,1].transAxes, fontweight='bold')
    
    ################################### Panel C
    plot_panel_c(b_drought_rain, axes[1,0])
    axes[1,0].text(-0.15, 1, 'c', fontsize=16, transform=axes[1,0].transAxes, fontweight='bold')

   # N = 6
   # hspace = 1.25
   # left = 0.1
   # 
   # ind = np.arange(3)/2.0 + left # the x locations for the groups
   # width = 0.45       # the width of the bars
   # 
   # # Alternative Red: "#e74c3c" and blue "#3498db"
   # 
   # #with sns.axes_style("whitegrid"):
   # rects1 = axes[1,0].bar(ind, drought_rain_table['mean'][0:3], width, color=colors[0], 
   #                        yerr=drought_rain_table['ci'][0:3], error_kw={'ecolor':'k','lw':2})
   # rects2 = axes[1,0].bar(ind + width + hspace,drought_rain_table['mean'][3:6], width, color=colors[-1], 
   #                        yerr=drought_rain_table['ci'][3:6], error_kw={'ecolor':'k','lw':3, 'capsize':0})
   # 
   # # add some text for labels, title and axes ticks
   # axes[1,0].set_ylabel('Yield change')
   # axes[1,0].set_title('Temperature interaction with drought and extreme rainfall')
   # # ax.set_xticks(ind + width / 2)
   # axes[1,0].set_xticks(np.concatenate((ind + width / 2, ind + width*1.5 + hspace)))
   # axes[1,0].set_xticklabels(('Dry(all)', 'Dry+hot', 'Dry-hot', 'Rain(all)','Rain+cold','Rain-cold'), 
   #                           fontsize=11)
   # # rotation=45, ha='right'
   # axes[1,0].set_xlim([ind[0]-left, ind[-1] + 2* width + hspace + 0.1])
   # 
   # axes[1,0].legend((rects1[0], rects2[0]), ('Drought', 'Extreme rainfall'), loc='upper left')

   # for i, line in enumerate(axes[1,0].lines):
   #     line.set_solid_capstyle('round')
   #     line.set_solid_joinstyle('round')
   #     line.set_dash_capstyle('round')
   #     line.set_dash_joinstyle('round')
    
    ################################# Panel D
    
    f2 = sns.barplot(x='Month of Loss Abbreviation', y='loss_ratio,weight', hue='Damage Cause Description', 
                     data=loss_temp, estimator=weighted_mean, order=['MAY','JUN','JUL','AUG'], 
                     hue_order=['Drought', 'Excess Moisture/Precip/Rain'],
                     palette=[colors[0],colors[-1]], saturation=1, ci=95, orient='v', ax=axes[1,1])
    
    axes[1,1].set_ylabel('Loss ratio', fontsize=12)
    axes[1,1].set_xticklabels(['May','June','July','August'])
    f2.set_xlabel('Month of loss', fontsize=12) 

    # change legend
    f2.legend_.set_title('Cause of loss')
    new_labels = ['Drought', 'Excess rain']
    for t, l in zip(f2.legend_.texts, new_labels): t.set_text(l)

    axes[1,1].text(-0.15, 1, 'd', fontsize=16, transform=axes[1,1].transAxes, fontweight='bold')

    
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.3)
    
   # plt.savefig('../figure/fig1_test4.png')
    plt.savefig('../figure/figure1.pdf')

if __name__ == "__main__":
    colors = define_colors()
    sns.set_style("ticks")
    make_plot()
