import pandas as pd
import xarray as xr
import statsmodels.api as sm
import numpy as np


def get_year_string(model_name, climate_data='agmerra'):
    if climate_data=='agmerra':
        s = {'cgms-wofost': [1980,2010],
             'clm-crop': [1980,2010],
             'epic-boku': [1980,2010],
             'epic-iiasa': [1980,2010],
             'gepic': [1980,2010],
             'lpj-guess': [1980,2010],
             'lpjml': [1980,2010],
             'orchidee-crop': [1980,2010],
             'papsim': [1980,2010],
             'pdssat': [1980,2010],
             'pegasus': [1980,2010],
             'pepic': [1980,2010]}
    else: # wfdei.gpcc
        s = {'cgms-wofost': [1979,2012],
             'clm-crop': [1980,2012],
             'epic-boku': [1979,2010],
             'epic-iiasa': [1979,2010],
             'gepic': [1979,2009],
             'lpj-guess': [1979,2009],
             'lpjml': [1979,2010],
             'orchidee-crop': [1979,2009],
             'papsim': [1979,2009],
             'pdssat': [1979,2009],
             'pegasus': [1979,2010],
             'pepic': [1979,2010]}
    return s[model_name]

"""
Load agmip data
Usage: ds = load_agmip_data(model_name, anomaly=False)
model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
               'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']
anomlay = True, load the detrended anomaly data from 1981 to 2010 
"""
def load_agmip_data(model_name, anomaly=False, climate_data='agmerra'):
    year_str = get_year_string(model_name, climate_data=climate_data)

    if climate_data == 'agmerra':
       year_end=2010
    else:
       year_end=2009

    if anomaly:
        f_dir = '~/Project/corn_extreme/data/result/'
        end_string = '_%s_hist_default_noirr_yield_mai_annual_anomaly_1981_%d.nc'%(climate_data,year_end)
    else:
        f_dir = '~/Project/data/agmip/maize/%s/'%climate_data
        end_string = '_%s_hist_default_noirr_yield_mai_annual_%d_%d.nc4'%(climate_data,year_str[0],year_str[1])
    
    fn = f_dir + model_name + end_string
    ds = xr.open_dataset(fn, decode_times=False)        
    return ds


"""
Calculate yield anomaly for model yield, return numpy array of the anomlay and trend term
Usage: array_ana, array_trend = get_yield_ana(ds_obs, d_model, obs_ana=False)
array_ana, array_trend = get_yield_ana(ds_obs, [], obs_ana=True)
when obs_ana = True, return yield_obs anamaly
"""
def get_yield_ana(ds_obs, d, obs_ana=False, climate_data='agmerra'):
    if climate_data == 'agmerra':
        year_end = 2010
        n_yr = 30
    else:
        year_end = 2009
        n_yr = 29 

    # Extract grid box where both obs and model data have values
    m1 = ds_obs['yield'].sum(axis=0).values
    
    if not obs_ana:
        m2 = d['yield_mai'].sum(axis=0).values
        mask = (m1!=0)&(m2!=0)
        idx = np.argwhere(mask)
        var_txt = 'yield_mai'
    else:
        mask = (m1!=0)
        idx = np.argwhere(mask)
        var_txt = 'yield'
        d = ds_obs.isel(time=range(0,n_yr))
        
    # Get slope and intercept for trend
    X = np.arange(1981,year_end+1)
    X = sm.add_constant(X)

    array_slope1 = np.zeros([360,720])
    array_intercept1 = np.zeros([360,720])

    for n in range(idx.shape[0]):
        lat_n = idx[n][0]
        lon_n = idx[n][1]

        y1 = d[var_txt][:,lat_n,lon_n].values

        mod_fit1 = sm.OLS(y1, X, missing='drop').fit()

        array_intercept1[lat_n,lon_n], array_slope1[lat_n,lon_n] = \
            mod_fit1.params[0], mod_fit1.params[1]
            
    
    # Get anomaly 
    array_year = np.zeros([n_yr,360,720])
    for y in range(1981,year_end+1):
        array_year[y-1981:,:] = y

    array_trend = np.zeros([n_yr,360,720])

    for y in range(1981,year_end+1):
        array_trend[y-1981:,:] = array_year[y-1981:,:] * array_slope1 + array_intercept1

    array_ana = d[var_txt].values - array_trend
    
    # Mast out non US region
    mask_3d = np.broadcast_to(mask, array_ana.shape)
    array_ana[~mask_3d] = np.nan
    array_trend[~mask_3d] = np.nan
    
    return array_ana, array_trend

"""
Save model yield anomaly from 1981 to 2010/2009 for US
"""
def save_agmip_anomaly(climate_data='agmerra'):
    ds_obs = xr.open_dataset('../data/result/corn_area_yield_1981_2010_05deg.nc')
    
    model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
                  'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']

    if climate_data=='agmerra':
        year_end = 2010
    else:
        year_end = 2009

    # saved netcdf file name
    end_string = '_%s_hist_default_noirr_yield_mai_annual_anomaly_1981_%d.nc'%(climate_data,year_end)

    for m in model_names:
        d = load_agmip_data(m,climate_data=climate_data)
        year_str = get_year_string(m, climate_data=climate_data) # get data year range

       # array_ana, array_trend = get_yield_ana(ds_obs, d.isel(time=range(1,31)))
        if climate_data=='agmerra':
            i1 = 1981 - year_str[0]
            i2 = 31
            array_ana, array_trend = get_yield_ana(ds_obs, d.isel(time=range(1,31)),climate_data=climate_data)
        else: # from 1981 to 2009 for gccp
            i1 = 1981 - year_str[0]
            i2 = 2009 - year_str[0] + 1
            array_ana, array_trend = get_yield_ana(ds_obs.isel(time=range(0,29)), d.isel(time=range(i1,i2)),climate_data=climate_data)

        # subset model data to create a new dataset
        ds = xr.Dataset({'yield_mai_ana': (['time', 'lat', 'lon'], array_ana),
                         'yield_mai_trend': (['time', 'lat', 'lon'], array_trend),
                         'yield_mai': (['time', 'lat', 'lon'], d.isel(time=range(i1,i2))['yield_mai'].values)
                               },
                        coords={'lon': d.lon,
                                'lat': d.lat,
                                'time': np.arange(1981,year_end+1,1)
                                 })

        ds.to_netcdf('../data/result/' + m + end_string)
        print('%s for %s saved'%(m,climate_data))

"""
Save 0.5 degree observed corn yield anomaly and area to netcdf file
"""
def save_yield_obs_anamaly(climate_data='agmerra'):
    ds_obs = xr.open_dataset('../data/result/corn_area_yield_1981_2010_05deg.nc')

    array_ana, array_trend = get_yield_ana(ds_obs, [], obs_ana=True)

    # subset model data to create a new dataset
    ds_obs_ana = xr.Dataset({'yield_ana': (['time', 'lat', 'lon'], array_ana),
                     'yield_trend': (['time', 'lat', 'lon'], array_trend),
                     'yield': (['time', 'lat', 'lon'], ds_obs['yield'].values),
                     'area': (['time', 'lat', 'lon'], ds_obs['area'].values)
                           },
                    coords={'lon': d.lon,
                            'lat': d.lat,
                            'time': np.arange(1981,2011,1)
                             })

    ds.to_netcdf('../data/result/corn_area_yield_anomaly_1981_2010_05deg.nc')
    print('corn_area_yield_anomaly_1981_2010_05deg.nc')


"""
Convert 3d xarry data array to pandas data frame by removeing na values
"""
def xarray2dataframe(da, name):
    return da.to_series().dropna().rename(name).to_frame().reset_index()

"""
Load agmip model data, match with obs, prec, and tmax bin and save to csv
"""
def agmip_to_csv(climate_data='agmerra'):
    model_names = ['obs','cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
                  'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']
    ds_obs = xr.open_dataset('../data/result/corn_area_yield_anomaly_1981_2010_05deg.nc')
   
    ds_obs['yield_ana_to_yield'] = ds_obs['yield_ana'] / ds_obs['yield_trend']

    rank = xr.open_dataset('../data/result/corn_Prec_Tmax_sigma_bin_1981_2010_05deg.nc')
    if climate_data == 'wfdei.gpcc':
        ds_obs = ds_obs.isel(time=range(0,29))
        rank = rank.isel(time=range(0,29))

    # prepare dataframe from obs
    t = xarray2dataframe(ds_obs['yield_ana_to_yield'], 'obs')
    t1 = xarray2dataframe(ds_obs['area'], 'Area')
    t2 = xarray2dataframe(rank['Prec_sigma_bin'], 'Prec_sigma_bin')
    t3 = xarray2dataframe(rank['Tmax_sigma_bin'], 'Tmax_sigma_bin')
    t = t.merge(t1, how='left').merge(t2, how='left').merge(t3, how='left')

    # Load model, covert and merge 
    for m in model_names[1::]:
        ds = load_agmip_data(m, anomaly=True, climate_data=climate_data)
        ds['yield_mai_ana_to_yield'] = ds['yield_mai_ana'] / ds['yield_mai_trend']
        t5 = xarray2dataframe(ds['yield_mai_ana_to_yield'], m)
#        t5 = ds['yield_mai_ana_to_yield'].to_series().rename(m).to_frame().rename(m)
        t = t.merge(t5, on=['lat','lon','time'], how='left')
     
    t.to_csv('../data/result/agmip_obs_yield_full_%s.csv'%climate_data, index=None)
    print('agmip for %s converted to csv, file saved'%climate_data)

if __name__ == '__main__':
#    save_agmip_anomaly(climate_data='wfdei.gpcc')
    agmip_to_csv(climate_data='wfdei.gpcc')
