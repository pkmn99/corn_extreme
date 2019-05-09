import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_figure1 import define_colors, weighted_mean
from plot_figure3 import column_weighted

climate_data = 'wfdei.gpcc'

if climate_data=='agmerra':
   data_tag = ''
else:
   data_tag = '_'+climate_data

agmip = pd.read_csv('../data/result/agmip_obs_yield_full%s.csv'%data_tag)
agmip.iloc[:,7::] = agmip.iloc[:,7::]*100


def mybartplot(model_name, ax, var='prec'):
    if var=='prec':
        binname='Prec_sigma_bin'
    if var=='tmax':
        binname='Tmax_sigma_bin'
     
    temp=agmip[[binname,model_name,'Area']].dropna()
    temp_obs=agmip[[binname,'obs','Area']].dropna()

    c0 = temp[model_name]<0
    p_neg = 100 * temp.loc[c0,:].groupby('Prec_sigma_bin').count()['Area'] \
              /temp.groupby('Prec_sigma_bin').count()['Area']

    p_neg.fillna(0).plot(color='g',legend=False,ax=ax,marker='.') # fill na with zero for ochidee

    c1 = temp_obs['obs']<0
    temp_obs=agmip[[binname,'obs','Area']].dropna()
    p_neg_obs = 100 * temp_obs.loc[c1,:].groupby('Prec_sigma_bin').count()['Area'] \
              /temp_obs.groupby('Prec_sigma_bin').count()['Area']

    p_neg_obs.plot(color='k',legend=False,ax=ax,marker='.')

    
    ax.set(xticks=np.arange(1.5,15.5,1), xticklabels=(x_txt))
    ax.set_title(model_name)
    ax.set_ylim(0,100)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xlim([1.5,15.5])
#


var = 'prec'
if var=='prec':
    binlabel='Precipitation anomaly ($\sigma$)'

model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
               'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']

fig, axes = plt.subplots(3,4,figsize=(16,9))

colors = define_colors()
x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
x_txt.insert(0,'')
x_txt.append('')

for i,m in enumerate(model_names):
    mybartplot(m, axes.flatten()[i],var=var)
    if i > 7:
        axes.flatten()[i].set(xticks=np.arange(1.5,15.5,1), xticklabels=(x_txt))
        axes.flatten()[i].set_xlabel(binlabel, labelpad=12)
        axes.flatten()[i].set_xticklabels(axes.flatten()[i].get_xticklabels(), fontsize=10, rotation=90)

    if i==4:
        axes.flatten()[i].set_ylabel('Percentage of negative yield impact (%)',fontsize=12)

legend_txt = ['Model', 'Obs']

axes.flatten()[0].legend(axes.flatten()[0].lines, (legend_txt),
              loc='upper center',frameon=False)
        
    
plt.savefig('../figure/figure_agmip_each_model_negimpact_%s.pdf'%climate_data)
print('figure saved')


