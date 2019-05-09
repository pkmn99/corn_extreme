import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Plot to show how drought and rainfall impacts depend on definition
"""

bin_yield = pd.read_csv('../data/result/bin_yield.csv', dtype={'FIPS':str})

sns.set_context("notebook")
sns.set_style("darkgrid")


# Prepare data
v_rain = np.zeros([8,3])
for k, i in enumerate(range(8,16)):
    c = bin_yield['Prec_sigma_bin']>=i 
    v_rain[k,0] = i
    v_rain[k,1] = (bin_yield[c].sum()['Yield_ana_to_yield_area']) / (bin_yield[c].sum()['Area']) * 100
    c2=bin_yield['Yield_ana_to_yield']<0
    v_rain[k,2] = (bin_yield[c&c2].sum()['Yield_ana_to_yield_area']) / (bin_yield[c&c2].sum()['Area']) * 100
    
    
v_dry = np.zeros([6,3])
for k, i in enumerate(range(2,8)):
    c = bin_yield['Prec_sigma_bin']<=i 
    v_dry[k,0] = i
    v_dry[k,1] = (bin_yield[c].sum()['Yield_ana_to_yield_area']) / (bin_yield[c].sum()['Area']) * 100
    c2=bin_yield['Yield_ana_to_yield']<0
    v_dry[k,2] = (bin_yield[c&c2].sum()['Yield_ana_to_yield_area']) / (bin_yield[c&c2].sum()['Area']) * 100    


# Begin plot

fig, [ax2,ax1] = plt.subplots(1,2, figsize=(10,4))

# Subplot 2 for rainfall    
x_txt_rain = ['','>0','>0.5','>1','>1.5','>2','>2.5','>3','>3.5']

ax1.plot(v_rain[:,0], v_rain[:,1::],'-o')
ax1.set_xlim(7.5,15.5)
ax1.set_ylim(-45,5)

ax1.set_xticklabels(x_txt_rain, rotation=0)
ax1.set_xlabel('Precipitation anomaly threshold ($\sigma$)')
ax1.set_ylabel('Yield change (%)')
ax1.text(-0.15, 1, 'b', fontsize=14, transform=ax1.transAxes, fontweight='bold')

ax1.text(0.5, 1.08, 'Excessive rainfall impact under different definitions', 
        transform=ax1.transAxes, ha='center',fontsize=11)
ax1.legend(['All samples','Negative only'],loc='lower left')

# Add text to all sample curve
for k, i in enumerate(v_rain[:,0]):
    if k == 5: # Use bold font to highlight
        ax1.text(i, v_rain[k,1]+1, np.round(v_rain[k,1],1), fontsize=12, fontweight='bold')
    else:    
        ax1.text(i, v_rain[k,1]+1, np.round(v_rain[k,1],1), fontsize=10)

for k, i in enumerate(v_rain[:,0]):
    ax1.text(i, v_rain[k,2]-3.5, np.round(v_rain[k,2],1), fontsize=10, ha='center')
    
    
# Subplot 1 for drought    
x_txt_dry = ['','<-2.5','<-2.0','<-1.5','<-1','<-0.5','<0']

ax2.plot(v_dry[:,0], v_dry[:,1::],'-o')
ax2.set_xlim(1.5,7.5)
ax2.set_ylim(-45,5)

ax2.set_xticklabels(x_txt_dry, rotation=0)
ax2.set_xlabel('Precipitation anomaly threshold ($\sigma$)')
ax2.set_ylabel('Yield change (%)')

ax2.text(-0.15, 1, 'a', fontsize=14, transform=ax2.transAxes, fontweight='bold')

ax2.text(0.5, 1.08, 'Extreme drought impact under different definitions', 
        transform=ax2.transAxes, ha='center',fontsize=11)
ax2.legend(['All samples','Negative only'],loc='lower right')

# Add text to all sample curve
for k, i in enumerate(v_dry[:,0]):
    if k == 1:
        ax2.text(i, v_dry[k,1]+1, np.round(v_dry[k,1],1), fontsize=12, ha='right',fontweight='bold')
    else:
        ax2.text(i, v_dry[k,1]+1, np.round(v_dry[k,1],1), fontsize=10, ha='right')

# Add text to negative only sample curve
for k, i in enumerate(v_dry[:,0]):
    ax2.text(i, v_dry[k,2]-3.5, np.round(v_dry[k,2],1), fontsize=10, ha='center')    

plt.subplots_adjust(bottom=0.12)
plt.savefig('../figure/figure_drought_rainfall_impact_definition.pdf')

