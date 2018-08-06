import numpy as np
import pandas as pd
import xarray as xr

"""
Load agmip data
Usage: ds = load_agmip_data(model_name, anomaly=False)
model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
               'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']
anomlay = True, load the detrended anomaly data from 1981 to 2010 
"""
def load_agmip_data(model_name, anomaly=False):
    if anomaly:
        f_dir = '~/Project/RMA_study/data/result/'
        end_string = '_agmerra_hist_default_noirr_yield_mai_annual_anomaly_1981_2010.nc'
    else:
        f_dir = '~/Project/data/agmip/maize/'
        end_string = '_agmerra_hist_default_noirr_yield_mai_annual_1980_2010.nc4'
    
    fn = f_dir + model_name + end_string
    ds = xr.open_dataset(fn, decode_times=False)        
    return ds


import statsmodels.api as sm

"""
Calculate yield anomaly for model yield, return numpy array of the anomlay and trend term
Usage: array_ana, array_trend = get_yield_ana(ds_obs, d_model, obs_ana=False)
array_ana, array_trend = get_yield_ana(ds_obs, [], obs_ana=True)
when obs_ana = True, return yield_obs anamaly
"""
def get_yield_ana(ds_obs, d, obs_ana=False):
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
        d = ds_obs
        
    # Get slope and intercept for trend
    X = np.arange(1981,2011)
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
    array_year = np.zeros([30,360,720])
    for y in range(1981,2011):
        array_year[y-1981:,:] = y

    array_trend = np.zeros([30,360,720])

    for y in range(1981,2011):
        array_trend[y-1981:,:] = array_year[y-1981:,:] * array_slope1 + array_intercept1

    array_ana = d[var_txt].values - array_trend
    
    # Mast out non US region
    mask_3d = np.broadcast_to(mask, array_ana.shape)
    array_ana[~mask_3d] = np.nan
    array_trend[~mask_3d] = np.nan
    
    return array_ana, array_trend

"""
Save model yield anomaly from 1981 to 2010 for US
"""
def save_agmip_anomaly():
    ds_obs = xr.open_dataset('../data/result/corn_area_yield_1981_2010_05deg.nc')
    
    model_names = ['cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
                  'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']
    # saved netcdf file name
    end_string = '_agmerra_hist_default_noirr_yield_mai_annual_anomaly_1981_2010.nc'

    for m in model_names:
        d = load_agmip_data(m)

        array_ana, array_trend = get_yield_ana(ds_obs, d.isel(time=range(1,31)))

        # subset model data to create a new dataset
        ds = xr.Dataset({'yield_mai_ana': (['time', 'lat', 'lon'], array_ana),
                         'yield_mai_trend': (['time', 'lat', 'lon'], array_trend),
                         'yield_mai': (['time', 'lat', 'lon'], d.isel(time=range(1,31))['yield_mai'].values)
                               },
                        coords={'lon': d.lon,
                                'lat': d.lat,
                                'time': np.arange(1981,2011,1)
                                 })

        ds.to_netcdf('../data/result/' + m + end_string)
        print('%s saved'%m)

"""
Save 0.5 degree observed corn yield anomaly and area to netcdf file
"""
def save_yield_obs_anamaly():
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
def agmip_to_csv():
    model_names = ['obs','cgms-wofost','clm-crop','epic-boku','epic-iiasa','gepic','lpj-guess',
                  'lpjml','orchidee-crop','papsim','pdssat','pegasus','pepic']
    ds_obs = xr.open_dataset('../data/result/corn_area_yield_anomaly_1981_2010_05deg.nc')

    ds_obs['yield_ana_to_yield'] = ds_obs['yield_ana'] / ds_obs['yield_trend']


    rank = xr.open_dataset('../data/result/corn_Prec_Tmax_sigma_bin_1981_2010_05deg.nc')

    # prepare dataframe from obs
    t = xarray2dataframe(ds_obs['yield_ana_to_yield'], 'obs')
    t1 = xarray2dataframe(ds_obs['area'], 'Area')
    t2 = xarray2dataframe(rank['Prec_sigma_bin'], 'Prec_sigma_bin')
    t3 = xarray2dataframe(rank['Tmax_sigma_bin'], 'Tmax_sigma_bin')
    t = t.merge(t1, how='left').merge(t2, how='left').merge(t3, how='left')

    # Load model, covert and merge 
    for m in model_names[1::]:
        ds = load_agmip_data(m, anomaly=True)
        ds['yield_mai_ana_to_yield'] = ds['yield_mai_ana'] / ds['yield_mai_trend']
        t5 = xarray2dataframe(ds['yield_mai_ana_to_yield'], m)
#        t5 = ds['yield_mai_ana_to_yield'].to_series().rename(m).to_frame().rename(m)
        t = t.merge(t5, on=['lat','lon','time'], how='left')
     
    t.to_csv('../data/result/agmip_obs_yield_full.csv', index=None)
    print('agmip converted to csv, file saved')

if __name__ == '__main__':
#    save_agmip_anomaly()
    agmip_to_csv()
