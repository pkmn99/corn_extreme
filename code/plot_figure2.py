# Plot figure 2, map and state bar_plot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

import pandas as pd
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

import seaborn as sns

from plot_figure1 import define_colors, weighted_mean

def norm_cmap(values, cmap, vmin=None, vmax=None):
    """
    Normalize and set colormap
    
    Parameters
    ----------
    values : Series or array to be normalized
    cmap : matplotlib Colormap
    normalize : matplotlib.colors.Normalize
    cm : matplotlib.cm
    vmin : Minimum value of colormap. If None, uses min(values).
    vmax : Maximum value of colormap. If None, uses max(values).
    
    Returns
    -------
    n_cmap : mapping of normalized values to colormap (cmap)
    
    """
#     mn = vmin or min(values)
#     mx = vmax or max(values)
#     norm = Normalize(vmin=mn, vmax=mx)
    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap, norm


# My own colormap or matplotlib default colormap
def my_colormap(name='RdBu', customize=False):
    # Combine from different colormaps
    if customize:
        red = plt.cm.RdBu(np.linspace(0., 0.5, 128))
        green = plt.cm.PRGn(np.linspace(0.5, 1, 128))
        blue = plt.cm.RdBu(np.linspace(1, 0.5, 128))

        if name == 'BuGr':
            # combine them and build a new colormap
            # https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
            colors = np.vstack((blue, green))
        if name == 'RdGr':
            colors = np.vstack((red, green))
            
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    else:
        cmap = plt.get_cmap(name)
    return cmap    

def plot_bar(bin_yield, ax, state='Illinois', xtick=True):
    x_txt = [str(i) for i in np.arange(-2.5,3.6,0.5)]
    x_txt.insert(0,'')
    x_txt.append('')
    
    c = bin_yield['State'] == state.upper()



    prec_bin_min = bin_yield[c].min()['Prec_sigma_bin']
    prec_bin_max = bin_yield[c].max()['Prec_sigma_bin']    
    x_offset = prec_bin_min-2

    with sns.axes_style("ticks"):
       # g = sns.barplot(x='Prec_sigma_bin', y='Yield_ana_to_yield', data=bin_yield[c], 
       #            palette=colors, ci=95, orient='v', saturation=1, 
       #            ax=ax)
        g = sns.barplot(x='Prec_sigma_bin', y='Yield_ana_to_yield,weight', estimator=weighted_mean,
                    data=bin_yield[c], palette=colors[x_offset::], ci=95, orient='v', saturation=1,
                    ax=ax)

        sns.despine()

    ax.set(xticks=np.arange(-0.5,14.5,1), xticklabels=x_txt)

    # move bars to adjust insufficient bins range on the left
    if x_offset>0:
        # Move bar
        for i, patch in enumerate(ax.patches):
            patch.set_x(patch.get_x() + x_offset)
    
        # Move error bar
        for i, line in enumerate(ax.lines):
            line.set_xdata(line.get_xdata() + x_offset)
    
    # option to plot xticklabel 
    if xtick:
        g.set_xticklabels(g.get_xticklabels(), fontsize=10, rotation=90)
    else:
        g.set_xticklabels([], rotation=90)
    
    ax.axes.tick_params(axis='y',labelsize=10)

    ax.set_ylim(-55,15)
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.text(0.5, 0.15, state, transform=ax.transAxes, fontsize=14, ha='center') #,fontweight='bold')

    # Calculte exteme drought and rainfall impact to show on bar chart
    c5 = bin_yield['Prec_sigma_bin']<4 # all drought

    v_drought = bin_yield[c&c5]['Yield_ana_to_yield_area'].sum()/bin_yield[c&c5]['Area'].sum()
    
    c6 = bin_yield['Prec_sigma_bin']>12 # all rain 
    v_rain = bin_yield[c&c6]['Yield_ana_to_yield_area'].sum()/bin_yield[c&c6]['Area'].sum()


    ax.text(0.02, 0.9, "{0:.1f}%".format(v_drought*100), transform=ax.transAxes,
                   fontsize=10, color=colors[0])

    ax.text(0.85, 0.9,"{0:.1f}%".format(v_rain*100), transform=ax.transAxes,
                   fontsize=10, color=colors[-1])

    # Change tick label color 
    [t.set_color(colors[0]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i<3]
    [t.set_color(colors[-1]) for i,t in enumerate(ax.xaxis.get_ticklabels()) if i>=11]


def plot_map(df, fig, ax, type='Drought'):
    if type == 'Drought':
        mycmap = my_colormap(name='RdGr', customize=True)

    if type == 'Rain':
        mycmap = my_colormap(name='BuGr', customize=True)
        
    cmap, norm = norm_cmap(df['Yield_ana_to_yield_weighted'], cmap=mycmap, 
                           vmin=-0.8, vmax=0.8)
    
    df['color'] = [cmap.to_rgba(value) for value in df['Yield_ana_to_yield_weighted'].values]
    
    ax.set_extent([-120, -73, 22, 50], ccrs.Geodetic())

    fips_list = df['FIPS'].tolist()

    # Plot county value    
    for record, county in list(zip(county_shapes.records(), county_shapes.geometries())):
        fips = record.attributes['FIPS']
        if fips in fips_list:
            facecolor = df[df['FIPS'] == fips]['color']
        else:
            facecolor = '#B3B3B3' #'grey'
        ax.add_geometries([county], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='white', linewidth=0)

    # Plot state boundary    
    for state in state_shapes.geometries():
        facecolor = 'None'
        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='black',linewidth=0.5)
    
#     
    ax.outline_patch.set_visible(False)
    cbar_height = 0.7 # percent of subplot height
    cbar_pos = [ax.get_position().x1-0.085, ax.get_position().y0 + ax.get_position().height*(1-cbar_height)/2.0,
                0.01, ax.get_position().height*cbar_height]
    cax = fig.add_axes(cbar_pos)

    cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=mycmap,
                                norm=norm, # label='Yield change (%)'
                                orientation='vertical',ticks=np.arange(-0.8, 0.9, 0.2))
    
    cb1.ax.set_yticklabels(np.arange(-80, 90, 20), fontsize=10) 
    cb1.set_label('Yield change (%)', fontsize=12)

# Customize x and y label for barplot
def set_x_y_label(ax, x=False, y=True):
    if x: 
        ax.set_xlabel('Precipitation anomaly ($\sigma$)',labelpad=30, fontsize=12)
        ax.text(0.0, -0.42, 'Extreme\ndry', transform=ax.transAxes, fontsize=10,
                                      color=colors[0])
        ax.text(0.25, -0.42, 'Moderate\ndry', transform=ax.transAxes, fontsize=10,
                       color=colors[2])
        ax.text(0.5, -0.42, 'Moderate\nwet', transform=ax.transAxes, fontsize=10,
                       color=colors[6])
        ax.text(0.75, -0.42, 'Extreme\n wet', transform=ax.transAxes, fontsize=10,
                       color=colors[-1])
    if y:
        ax.set_ylabel('Yield change (%)', fontsize=12)


def make_plot():

    #plt.close("all")
    
    fig = plt.figure(figsize=(15,12))
    
    gs = GridSpec(6, 4)
    
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[0,2])
    ax3 = plt.subplot(gs[1,-1])
    ax4 = plt.subplot(gs[2,-1])
    ax5 = plt.subplot(gs[3,-1])
    ax6 = plt.subplot(gs[4,-1])
    ax7 = plt.subplot(gs[-1,-2])
    ax8 = plt.subplot(gs[-1,1])
    ax9 = plt.subplot(gs[-2,0])
    ax10 = plt.subplot(gs[-3,0])
    ax11 = plt.subplot(gs[-4,0])
    ax12 = plt.subplot(gs[-5,0])
    
    # plot bars 
    plot_bar(bin_yield, ax1, state='Minnesota',xtick=False)
    set_x_y_label(ax1)
    
    plot_bar(bin_yield, ax2, state='Wisconsin',xtick=False)
    plot_bar(bin_yield, ax3, state='Michigan',xtick=False)
    set_x_y_label(ax3,y=False)
    
    plot_bar(bin_yield, ax4, state='Illinois',xtick=False)
    set_x_y_label(ax4,y=False)
    
    plot_bar(bin_yield, ax5, state='Indiana',xtick=False)
    set_x_y_label(ax5,y=False)
    
    plot_bar(bin_yield, ax6, state='Ohio')
    set_x_y_label(ax6, x=True,y=False)

    plot_bar(bin_yield, ax7, state='Missouri')
    set_x_y_label(ax7, x=True, y=False)
    ax7.set_zorder(10)
    
    plot_bar(bin_yield, ax8, state='Kansas')
    set_x_y_label(ax8, x=True)
    ax8.set_zorder(10)
    
    
    plot_bar(bin_yield, ax9, state='Nebraska')
    set_x_y_label(ax9, x=True)
    
    plot_bar(bin_yield, ax10, state='Iowa',xtick=False)
    set_x_y_label(ax10)
    
    plot_bar(bin_yield, ax11, state='South Dakota',xtick=False)
    set_x_y_label(ax11)
    
    plot_bar(bin_yield, ax12, state='North Dakota',xtick=False)
    set_x_y_label(ax12)

    # Add panel label 
    l = [chr(i) for i in range(ord('c'), ord('n')+1)]
    for i, axx in enumerate ([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]):
        axx.text(-0.15, 1, l[i], fontsize=16, transform=axx.transAxes, fontweight='bold')


    
    # ax_m2 = plt.subplot(gs[3:5, 1:3], projection=ccrs.LambertConformal()) # Map 2
    
    # Use customized axes to make the map larger, based on ax_m2 by gridspec 
    new_pos1 = [0.32717391304347831*0.55, 0.52357142857142858*0.975, 0.3706521739130435*1.5, 0.24357142857142855*1.2]
    new_ax1 = plt.axes(new_pos1, projection=ccrs.LambertConformal())
    plot_map(drought_value, fig, new_ax1)
    new_ax1.text(0.55,0.925,'Extreme drought', transform=new_ax1.transAxes, fontsize=14, fontweight='bold',
                 color=colors[0])

    new_ax1.text(0.0, 1, 'a', fontsize=16, transform=new_ax1.transAxes, fontweight='bold')
    
    new_pos2 = [0.32717391304347831*0.55, 0.25785714285714278*0.9, 0.3706521739130435*1.5, 0.24357142857142866*1.2]
    new_ax2 = plt.axes(new_pos2, projection=ccrs.LambertConformal())
    plot_map(rain_value, fig, new_ax2, type='Rain')
    new_ax2.text(0.55,0.925,'Excessive rainfall', transform=new_ax2.transAxes,fontsize=14, fontweight='bold',
                color=colors[-1])
    new_ax2.patch.set_visible(False)
    new_ax2.text(0.0, 1, 'b', fontsize=16, transform=new_ax2.transAxes, fontweight='bold')
    
    
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.1)

    
    plt.savefig('../figure/figure2_new.pdf')


if __name__ == "__main__":
    colors = define_colors()

    # Load data
    shapefile='../data/US_county_gis/counties_contiguous.shp'
    county_shapes = shpreader.Reader(shapefile)
    shapefile='../data/US_county_gis/states_contiguous.shp'
    state_shapes = shpreader.Reader(shapefile)
    
    bin_yield = pd.read_csv('../data/result/bin_yield.csv', dtype={'FIPS':str})
    bin_yield['Yield_ana_to_yield_area'] = bin_yield['Yield_ana_to_yield'] * bin_yield['Area']
    bin_yield['Yield_ana_to_yield,weight'] = list(zip(bin_yield['Yield_ana_to_yield']*100, bin_yield['Area']))
    
    # Drought impact map
    c1 = bin_yield['Prec_sigma_bin']<4
    temp = bin_yield[c1][['State','FIPS','Yield_ana','Yield_ana_to_yield','Yield_ana_to_yield_area','Area']]
    gp = temp.groupby('FIPS')
    drought_value = (gp.sum()['Yield_ana_to_yield_area']/gp.sum()['Area']).to_frame('Yield_ana_to_yield_weighted').reset_index()
    
    # Rain impact map
    c1 = bin_yield['Prec_sigma_bin']>12
    temp = bin_yield[c1][['State','FIPS','Yield_ana','Yield_ana_to_yield','Yield_ana_to_yield_area','Area']]
    gp = temp.groupby('FIPS')
    rain_value = (gp.sum()['Yield_ana_to_yield_area']/gp.sum()['Area']).to_frame('Yield_ana_to_yield_weighted').reset_index()

    make_plot()
