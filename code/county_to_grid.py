import numpy as np
import pandas as pd
import xarray as xr
from load_nass_data import load_nass_county_data
from get_county_latlon import get_county_latlon


# functions related to aggregating county yield/area to 0.5 deg grid
# Run with geo env

"""
Get gridded corn harvest area and weighted yield from 1981 to 2010
and save results to netcdf file
"""
def get_grid_yield_area(rerun=False):
    if rerun: 
        # Load county information
        county_info = get_county_latlon()
        
        # Load corn yield data
        corn_yield = load_nass_county_data('corn', 'grain_yield', 'allstates', 1981, 2016)
        corn_yield.rename(columns={'Value':'Yield'}, inplace=True)

        corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1981, 2016)
        corn_area.rename(columns={'Value':'Area'}, inplace=True)
        corn_combined = corn_yield.dropna().merge(corn_area[['Year','FIPS','Area']].dropna(),
                                                  on=['FIPS','Year'], left_index=True)

        corn_combined = corn_combined.merge(county_info, on='FIPS')
        
        # model output from 1980 to 2010, we only need 1981 to 2010
        array_area = np.zeros([360,720,30])
        array_yield = np.zeros([360,720,30])

        for y in range(1981, 2011):
            temp = corn_combined[corn_combined['Year']==y]
            for n in range(temp.shape[0]):
                r = temp.iloc[n,:]['row']
                c = temp.iloc[n,:]['col']
                array_area[r,c,y-1981] = array_area[r,c,y-1981] + temp.iloc[n,:]['Area']
                array_yield[r,c,y-1981] = array_yield[r,c,y-1981] + temp.iloc[n,:]['Area']*temp.iloc[n,:]['Yield'] 

        array_yield = array_yield/array_area

        # Save to netcdf 
        lon = np.arange(-179.75,180,0.5)
        lat = np.arange(89.75,-90,-0.5)
        ds = xr.Dataset({'area': (['time','lat', 'lon'], np.transpose(array_area, (2, 0, 1))),
                              'yield': (['time','lat', 'lon'], np.transpose(array_yield, (2, 0, 1)))
                             },
                        coords={'lon': lon,
                                'lat': lat,
                                'time': np.arange(1981,2011,1)
                                 })
        ds.to_netcdf('../data/result/corn_area_yield_1981_2010_05deg.nc')
        print('Rerun. File corn_area_yield_1981_2010_05deg.nc saved!')
    else:
        print('Load file corn_area_yield_1981_2010_05deg.nc!')
        ds = xr.open_dataset('../data/result/corn_area_yield_1981_2010_05deg.nc')
    return ds

"""
Aggregate the county level climate extreme from Prism to 0.5 degree
Usage: get_grid_extreme_climate_rank(rerun=False)
"""
def get_grid_extreme_climate_rank(rerun=False):
    bin_yield = pd.read_csv('../data/result/bin_yield.csv',dtype={'FIPS':str})
    county_info = pd.read_csv('../data/result/county_latlon.csv',dtype={'FIPS':str})
    bin_yield = bin_yield.merge(county_info, on='FIPS')

    array_Prec_sigma_bin = np.zeros([360,720,30])
    array_Tmax_sigma_bin = np.zeros([360,720,30])

    array_area = np.zeros([360,720,30])

    for y in range(1981, 2011):
        temp = bin_yield[bin_yield['Year']==y]
        for n in range(temp.shape[0]):
            r = temp.iloc[n,:]['row']
            c = temp.iloc[n,:]['col']
            array_area[r,c,y-1981] = array_area[r,c,y-1981] + temp.iloc[n,:]['Area']
            array_Prec_sigma_bin[r,c,y-1981] = array_Prec_sigma_bin[r,c,y-1981] + \
                temp.iloc[n,:]['Prec_sigma_bin']*temp.iloc[n,:]['Area'] 
            array_Tmax_sigma_bin[r,c,y-1981] = array_Tmax_sigma_bin[r,c,y-1981] + \
                temp.iloc[n,:]['Tmax_sigma_bin']*temp.iloc[n,:]['Area'] 

    array_Prec_sigma_bin = array_Prec_sigma_bin/array_area
    array_Tmax_sigma_bin = array_Tmax_sigma_bin/array_area

    array_Prec_sigma_bin = np.round(array_Prec_sigma_bin)
    array_Tmax_sigma_bin = np.round(array_Tmax_sigma_bin)

    # Save to netcdf 
    lon = np.arange(-179.75,180,0.5)
    lat = np.arange(89.75,-90,-0.5)
    ds = xr.Dataset({'Prec_sigma_bin': (['time','lat', 'lon'], np.transpose(array_Prec_sigma_bin, (2, 0, 1))),
                     'Tmax_sigma_bin': (['time','lat', 'lon'], np.transpose(array_Tmax_sigma_bin, (2, 0, 1)))
                         },
                    coords={'lon': lon,
                            'lat': lat,
                            'time': np.arange(1981,2011,1)
                             })
    ds.to_netcdf('../data/result/corn_Prec_Tmax_sigma_bin_1981_2010_05deg.nc')
    print('File corn_Prec_Tmax_sigma_bin_1981_2010_05deg.nc saved!')

if __name__ == '__main__':
#    get_grid_yield_area(rerun=True)
    get_grid_extreme_climate_rank(rerun=True)

