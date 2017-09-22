import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

"""
Load NASS data from csv file, choosing crop, variable, state
df = load_nass_data(crop, variable, state_name, start_year, end_year)
crop: 'corn', 'soybean'
variable: corn: 'grain_areaharvested', 'grain_yield', 'condition'
          soybeans: 'yield', 'areaharvested', 'condition', areaplanted
"""
# 04/04/2017 initial version
# 06/06 read allstates data

def file_year_txt(var):
    var_year_txt = '1980-2016'
    if var == 'condition': 
        var_year_txt = '1986-2016'
    return var_year_txt

def file_level_txt(var):
    level = 'county'
    if var in ['condition','progress']: 
        level = 'state'
    return level 

def var_col_txt(var):
    cols = ['Year','State', 'State ANSI', 'County',
            'County ANSI','Commodity','Data Item','Value']
    if var in ['condition','progress']: 
        cols = ['Year','Week Ending','State', 'State ANSI',
                'County ANSI','Commodity','Data Item','Value']
    return cols 


def load_nass_county_data(crop, variable, state_name, start_year, end_year):
    
    # custumized csv_read
    def my_read_csv(csv_filename, cols):
        df = pd.read_csv(csv_filename, usecols=cols, thousands=',', 
                           dtype={'State ANSI':str, 'County ANSI':str})
        return df
        
    path_file = '~/Project/data/NASS/'
    level = file_level_txt(variable)

    # coloums to load
    cols = var_col_txt(variable)
        
    # If Illinois, directly use Illinois data file, otherwise, extract state from
    # all states file
    if state_name == 'Illinois':
        csv_filename = ('%s%s_%s_%s_%s_%s.csv' %(path_file,crop.upper(), 
                                                        variable.upper(), 
                                                        state_name.upper(), 
                                                        level.upper(),
                                                        file_year_txt(variable)))
        data = my_read_csv(csv_filename, cols)
        data = data[(data['Year']>=start_year) \
                        & (data['Year']<=end_year)].copy()
        
    elif state_name == 'allstates':
        if variable in ['grain_areaharvested', 'grain_yield', 'condition',
                'yield', 'areaharvested','areaplanted']:
            csv_filename1 = ('%s%s_%s_ALLSTATES_%s_%s_part1.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable)))
            csv_filename2 = ('%s%s_%s_ALLSTATES_%s_%s_part2.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable)))
            data_raw1 = my_read_csv(csv_filename1, cols)
            data_raw2 = my_read_csv(csv_filename2, cols)
            data_raw = pd.concat([data_raw1, data_raw2])

        else: 
            csv_filename = ('%s%s_%s_ALLSTATES_%s_%s.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable)))
            data_raw = my_read_csv(csv_filename, cols)
        
        data = data_raw[(data_raw['Year']>=start_year) \
                        & (data_raw['Year']<=end_year)].copy()
        
    data['FIPS'] = data['State ANSI'] + data['County ANSI']

    return data 


#"""
#Get detrended yield for each county 08/28/17
#Usage: yield_sample, trend_para = get_yield_anomaly(corn_yield)
#"""
#def get_yield_anomaly(corn_yield):
#    # format the yield data
#    yield_sample = corn_yield[['FIPS','Year','Yield','State']].dropna().set_index(['FIPS','Year'])
#    s = yield_sample.unstack('FIPS')['Yield'].shape  # size, year by FIPS
#    B = np.zeros([s[1],3]) # a,b,n, three parameters for linear trend, intercept, slope, and sample number
#
#    # estimate linear trend for each column (FIPS)
#    for i in range(s[1]):
#        temp = yield_sample.unstack('FIPS')['Yield'].iloc[:,i].to_frame('Yield').reset_index()
#        mod_fit = smf.ols(formula="Yield ~ Year", data=temp).fit()
#        B[i,0],B[i,1],B[i,2]= mod_fit.params[0], mod_fit.params[1], temp['Yield'].dropna().shape[0]
#
#    trend_para = pd.DataFrame(B, index=yield_sample.unstack('FIPS')['Yield'].columns, \
#                              columns=['intercept','slope','N'])
#    
#    yield_ana_sample = yield_sample.unstack('FIPS')['Yield'].copy()
#
#    # get anomaly by array multiplication through broadcasting
#    year_start = yield_sample.index.get_level_values(1).min()
#    year_end = yield_sample.index.get_level_values(1).max()
#    num_year = year_end - year_start + 1
#    
#    ana = yield_ana_sample.values - \
#        np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
#        * np.array([trend_para.T.loc['slope'].values,] * num_year) \
#        - np.array([trend_para.T.loc['intercept'].values,] * num_year) \
#
#    yield_ana_sample.iloc[:,:] = ana
#    # append anomaly to yield data
#    yield_sample = yield_sample.reset_index().merge(yield_ana_sample.stack().reset_index().rename(columns={0:'Yield_ana'}))
#    
#    return yield_sample, trend_para

"""
Detrended yield for each county 08/28/17
Add Harvested area anomaly 09/08
Usage: corn_sample, trend_para = add_yield_anomaly(corn_combined)
"""
def add_yield_anomaly(corn_combined, rerun=False):
    if rerun:
        # format the yield data
        combined_sample = corn_combined[['FIPS','Year','Yield','Area','State']].dropna().set_index(['FIPS','Year'])
        s = combined_sample.unstack('FIPS')['Yield'].shape  # size, year by FIPS
        B = np.zeros([s[1],6]) # a,b,n, three parameters (intercept, slope, and sample number) for linear trend of yield and area, 

        # estimate linear trend for each column (FIPS)
        for i in range(s[1]):
            # First make yield anomaly
            temp = combined_sample.unstack('FIPS')['Yield'].iloc[:,i].to_frame('Yield').reset_index()
            mod_fit = smf.ols(formula="Yield ~ Year", data=temp).fit()
            B[i,0],B[i,1],B[i,2]= mod_fit.params[0], mod_fit.params[1], temp['Yield'].dropna().shape[0]
            
            # Second make area anomaly
            temp2 = combined_sample.unstack('FIPS')['Area'].iloc[:,i].to_frame('Area').reset_index()
            mod_fit2 = smf.ols(formula="Area ~ Year", data=temp2).fit()
            B[i,3],B[i,4],B[i,5]= mod_fit2.params[0], mod_fit2.params[1], temp2['Area'].dropna().shape[0]
            

        trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
                                  columns=['Yield_intercept','Yield_slope','Yield_N',
                                          'Area_intercept','Area_slope','Area_N'])
        
        yield_ana_sample = combined_sample.unstack('FIPS')['Yield'].copy()
        area_ana_sample = combined_sample.unstack('FIPS')['Area'].copy()


        # get anomaly by array multiplication through broadcasting
        year_start = combined_sample.index.get_level_values(1).min()
        year_end = combined_sample.index.get_level_values(1).max()
        num_year = year_end - year_start + 1
        
        array_yield_ana = yield_ana_sample.values - \
            np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
            * np.array([trend_para.T.loc['Yield_slope'].values,] * num_year) \
            - np.array([trend_para.T.loc['Yield_intercept'].values,] * num_year)
        
        yield_ana_sample.iloc[:,:] = array_yield_ana
        
            
        array_area_ana = area_ana_sample.values - \
            np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
            * np.array([trend_para.T.loc['Area_slope'].values,] * num_year) \
            - np.array([trend_para.T.loc['Area_intercept'].values,] * num_year)    
            
        area_ana_sample.iloc[:,:] = array_area_ana
        
        # append anomaly to yield data
        combined_sample = combined_sample.reset_index(). \
            merge(yield_ana_sample.stack().reset_index().rename(columns={0:'Yield_ana'})).\
            merge(area_ana_sample.stack().reset_index().rename(columns={0:'Area_ana'}))
        # save for reuse
        combined_sample.to_csv('../data/result/corn_yield_area_anomaly.csv', index=False)
        trend_para.to_csv('../data/result/corn_yield_area_trend_para.csv', index=False)
        print 'file saved to ../data/result'
    else:
        print 'Load variable from saved files'
        combined_sample = pd.read_csv('../data/result/corn_yield_area_anomaly.csv',dtype={'FIPS':object})
        trend_para = pd.read_csv('../data/result/corn_yield_area_trend_para.csv')

    return combined_sample, trend_para
