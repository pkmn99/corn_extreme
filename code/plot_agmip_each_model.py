import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_figure1 import define_colors, weighted_mean
from plot_figure3 import column_weighted


agmip = pd.read_csv('../data/result/agmip_obs_yield_full.csv')
agmip.iloc[:,7::] = agmip.iloc[:,7::]*100


def mybartplot(model_name, ax):
     
    temp=agmip[['Prec_sigma_bin',model_name,'Area']].dropna()
    temp[model_name+',weight'] = zip(temp[model_name],temp['Area'])

#    sns.barplot(x='Prec_sigma_bin', y=model_name+',weight', estimator=weighted_mean,
#                data=temp, palette=colors, ci=95, orient='v', saturation=1,
#                ax=ax)
    
    sns.barplot(x='Prec_sigma_bin', y=model_name,
            data=agmip, palette=colors, ci=95, orient='v', saturation=1,
            ax=ax)
    
    ax.set(xticks=np.arange(-0.5,14.5,1), xticklabels=(x_txt))
    ax.set_title(model_name)
    ax.set_ylim(-60,60)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')


model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
               'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']

fig, axes = plt.subplots(3,4,figsize=(16,9))

colors = define_colors()
x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
x_txt.insert(0,'')
x_txt.append('')


for i,m in enumerate(model_names):
    mybartplot(m, axes.flatten()[i])
    if i > 7:
        axes.flatten()[i].set(xticks=np.arange(-0.5,14.5,1), xticklabels=(x_txt))
        axes.flatten()[i].set_xlabel('Precipitation anomaly ($\sigma$)', labelpad=12)
        axes.flatten()[i].set_xticklabels(axes.flatten()[i].get_xticklabels(), fontsize=10, rotation=90)

    if i%4==0:
        axes.flatten()[i].set_ylabel('Yield change (%)')    
    
plt.savefig('../figure/figure_agmip_each_model_prec.pdf')


