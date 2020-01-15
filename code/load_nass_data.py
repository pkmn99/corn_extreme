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

# Get end year for file of different crop types 
def get_end_year(crop_type):
    s = {'tomatoes':2017,'potatoes':2016,'sweetcorn':2013}
    return s[crop_type]

def file_year_txt(var,crop='corn'):
    if crop == 'corn':
        var_year_txt = '1980-2016'
    else:
        end_year = get_end_year(crop)
        var_year_txt = '1981-%d'%end_year

    if var == 'condition': 
        var_year_txt = '1986-2016'
    return var_year_txt

def file_level_txt(var):
    level = 'county'
    if var in ['condition','progress','grain_population']: 
        level = 'state'
    return level 

def var_col_txt(var):
    cols = ['Year','State', 'State ANSI', 'County',
            'County ANSI','Commodity','Data Item','Value']
    if var in ['condition','progress']: 
        cols = ['Year','Week Ending','State', 'State ANSI',
                'County ANSI','Commodity','Data Item','Value']
    return cols 


def load_nass_county_data(crop, variable, state_name, start_year, end_year,level='county'):
    
    # custumized csv_read
    def my_read_csv(csv_filename, cols):
        df = pd.read_csv(csv_filename, usecols=cols, thousands=',', 
                dtype={'State ANSI':str, 'County ANSI':str})
        return df

    path_file = '/media/liyan/HDD/Project/data/NASS/'
    if level=='county':    
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
        
    elif (state_name == 'allstates'):
        if (variable in ['grain_areaharvested', 'grain_yield', 'condition',
                'yield', 'areaharvested','areaplanted'])&(crop=='corn'):
            csv_filename1 = ('%s%s_%s_ALLSTATES_%s_%s_part1.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable,crop=crop)))
            csv_filename2 = ('%s%s_%s_ALLSTATES_%s_%s_part2.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable,crop=crop)))
            data_raw1 = my_read_csv(csv_filename1, cols)
            data_raw2 = my_read_csv(csv_filename2, cols)
            data_raw = pd.concat([data_raw1, data_raw2])

        else: 
            csv_filename = ('%s%s_%s_ALLSTATES_%s_%s.csv' %(path_file,
                                                               crop.upper(),
                                                               variable.upper(), 
                                                               level.upper(),
                                                               file_year_txt(variable,crop=crop)))
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
def add_yield_anomaly(corn_combined, rerun=False, fitting_type='linear'):

    if rerun:
        # format the yield data
        combined_sample = corn_combined[['FIPS','Year','Yield','Area','State']].dropna().set_index(['FIPS','Year'])
        s = combined_sample.unstack('FIPS')['Yield'].shape  # size, year by FIPS


        if fitting_type=='linear':
            formula_txt = "Yield ~ Year"
            B = np.zeros([s[1],6]) # a,b,n, three parameters (intercept, slope, and sample number) for linear trend of yield and area, 
            trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
                                  columns=['Yield_intercept','Yield_slope','Yield_N',
                                          'Area_intercept','Area_slope','Area_N'])
        if fitting_type=='quadratic':
            formula_txt = "Yield ~ Year + np.power(Year, 2)"
            B = np.zeros([s[1],7]) # a,b,n, four parameters (intercept, slope_1,slope_2, and sample number) for linear trend of yield and area, 
            trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
                                  columns=['Yield_intercept','Yield_slope1','Yield_slope2','Yield_N',
                                          'Area_intercept','Area_slope','Area_N'])


        # estimate linear trend for each column (FIPS)
        for i in range(s[1]):
            # First make yield anomaly
            temp = combined_sample.unstack('FIPS')['Yield'].iloc[:,i].to_frame('Yield').reset_index()
            mod_fit = smf.ols(formula=formula_txt, data=temp).fit()
            if fitting_type == 'linear':
                B[i,0],B[i,1],B[i,2]= mod_fit.params[0], mod_fit.params[1], temp['Yield'].dropna().shape[0]
            if fitting_type=='quadratic':
                B[i,0],B[i,1],B[i,2], B[i,3]=mod_fit.params[0],mod_fit.params[1],mod_fit.params[2],temp['Yield'].dropna().shape[0]
            
            # Second make area anomaly
            temp2 = combined_sample.unstack('FIPS')['Area'].iloc[:,i].to_frame('Area').reset_index()
            mod_fit2 = smf.ols(formula="Area ~ Year", data=temp2).fit()
            if fitting_type == 'linear':
                B[i,3],B[i,4],B[i,5]= mod_fit2.params[0], mod_fit2.params[1], temp2['Area'].dropna().shape[0]
            if fitting_type=='quadratic':
                B[i,4],B[i,5],B[i,6]= mod_fit2.params[0], mod_fit2.params[1], temp2['Area'].dropna().shape[0]
            

    #    if fitting_type=='linear':
    #        trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
    #                              columns=['Yield_intercept','Yield_slope','Yield_N',
    #                                      'Area_intercept','Area_slope','Area_N'])
    #    if fitting_type=='quadratic':
    #        trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
    #                              columns=['Yield_intercept','Yield_slope1','Yield_slope2','Yield_N',
    #                                      'Area_intercept','Area_slope','Area_N'])
    #
        yield_ana_sample = combined_sample.unstack('FIPS')['Yield'].copy()
        area_ana_sample = combined_sample.unstack('FIPS')['Area'].copy()


        # get anomaly by array multiplication through broadcasting
        year_start = combined_sample.index.get_level_values(1).min()
        year_end = combined_sample.index.get_level_values(1).max()
        num_year = year_end - year_start + 1

        if fitting_type == 'linear':
            array_yield_ana = yield_ana_sample.values - \
                np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
                * np.array([trend_para.T.loc['Yield_slope'].values,] * num_year) \
                - np.array([trend_para.T.loc['Yield_intercept'].values,] * num_year)
        if fitting_type=='quadratic':
            array_yield_ana = yield_ana_sample.values - \
                np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
                * np.array([trend_para.T.loc['Yield_slope1'].values,] * num_year) \
                - np.power(np.array([np.arange(year_start, year_end + 1),] * s[1]).T, 2) \
                * np.array([trend_para.T.loc['Yield_slope2'].values,] * num_year) \
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
        combined_sample.to_csv('../data/result/corn_yield_area_anomaly_%s.csv'%fitting_type, index=False)
        trend_para.to_csv('../data/result/corn_yield_area_trend_para_%s.csv'%fitting_type, index=False)
        print('file saved to ../data/result')
    else:
        print('Load variable from saved files')
        combined_sample = pd.read_csv('../data/result/corn_yield_area_anomaly_%s.csv'%fitting_type,dtype={'FIPS':object})
        trend_para = pd.read_csv('../data/result/corn_yield_area_trend_para_%s.csv'%fitting_type)

    return combined_sample, trend_para

# Get the corn progress for each month
def get_corn_progress():    
    # Load progress (state) and corn area (county)
    corn_progress= load_nass_county_data('corn', 'progress', 'allstates', 1981, 2016)

    corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1981, 2016)
    corn_area.rename(columns={'Value':'Area'}, inplace=True)

    # Covert time string to pd.Period 
    toweek = lambda x: pd.Period(x, freq='W')
    corn_progress['Time'] = corn_progress['Week Ending'].map(toweek)

    # Add month for each week
    corn_progress['Month'] = corn_progress['Time'].map(lambda x: x.month)

    # Merge with area
    corn_progress = corn_progress.merge(corn_area.groupby(['State','Year']).sum()['Area'].reset_index(),
                                        on=['State','Year'])

    # Weighed by Area
    corn_progress['Value_area'] = corn_progress['Value'] * corn_progress['Area']

    c = corn_progress['Year'] >1980
    result = corn_progress[c].groupby(['Month','Data Item']).sum()['Value_area'] / \
             corn_progress[c].groupby(['Month']).sum()['Value_area']
        
    result_area = corn_progress[c].groupby('Month').sum()['Value_area']
    
    return result


"""
Get historical irrigation area percentage
"""
def irrigation_percent(level='State'):
    # Load area
    corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1981, 2016)
    corn_area.rename(columns={'Value':'Area'}, inplace=True)
    corn_area.dropna(inplace=True)

    corn_irr_area = load_nass_county_data('corn', 'grain_irrigated_areaharvested', 'allstates', 1981, 2016)
    corn_irr_area.rename(columns={'Value':'Irr_Area'}, inplace=True)
    corn_irr_area.dropna(inplace=True)
    area_combined = corn_area.merge(corn_irr_area[['FIPS','Year','Irr_Area']], on=['FIPS','Year'])
    
    return (area_combined.groupby(level).sum()['Irr_Area']/corn_area.groupby(level).sum()['Area']).dropna()


"""
Save NASS data for crop modeling purpose
"""
def save_nass_yield_area(crop_type='corn'):
    head_txt = {'corn':'grain_','soybean':''}
    # Load harvest area 
    corn_area = load_nass_county_data(crop_type, '%sareaharvested'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_area.rename(columns={'Value':'area'}, inplace=True)
    corn_area.dropna(inplace=True)
    
    corn_area_irr = load_nass_county_data(crop_type, '%sirrigated_areaharvested'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_area_irr.rename(columns={'Value':'area_irr'}, inplace=True)
    
    corn_area_noirr = load_nass_county_data(crop_type, '%snonirrigated_areaharvested'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_area_noirr.rename(columns={'Value':'area_noirr'}, inplace=True)
    
    # Load yield data
    corn_yield = load_nass_county_data(crop_type, '%syield'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_yield.rename(columns={'Value':'yield'}, inplace=True)
    
    corn_yield_irr = load_nass_county_data(crop_type, '%sirrigated_yield'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_yield_irr.rename(columns={'Value':'yield_irr'}, inplace=True)
    
    corn_yield_noirr = load_nass_county_data(crop_type, '%snonirrigated_yield'%head_txt[crop_type], 'allstates', 1981, 2016)
    corn_yield_noirr.rename(columns={'Value':'yield_noirr'}, inplace=True)

    # Combine yield and harvest area
    df_final = corn_yield[['Year','FIPS','County','State','yield']].dropna().merge(corn_yield_irr[['Year','FIPS','yield_irr']],
                                                            on=['Year','FIPS'],how='left') \
                                                     .merge(corn_yield_noirr[['Year','FIPS','yield_noirr']],
                                                            on=['Year','FIPS'],how='left') \
                                                     .merge(corn_area[['Year','FIPS','area']],
                                                            on=['Year','FIPS'],how='left') \
                                                     .merge(corn_area_irr[['Year','FIPS','area_irr']],
                                                            on=['Year','FIPS'],how='left') \
                                                     .merge(corn_area_noirr[['Year','FIPS','area_noirr']],
                                                            on=['Year','FIPS'],how='left')
    df_final.rename(columns={'Year':'year'}, inplace=True)
    df_final.to_csv('../../crop_modeling/data/nass_yield_area_1981_2016_%s.csv'%crop_type, index=False)
    print('NASS data for %s saved to csv file in the crop modeling data folder'%crop_type)



"""
Save NASS F&V data for crop modeling purpose
"""
def save_nass_yield_area_FV(crop_type='corn', level='state'):
    end_year = get_end_year(crop_type)

    # Load yield data
    corn_yield = load_nass_county_data(crop_type, 'yield', 'allstates', 1981, end_year,level=level)
    corn_yield.rename(columns={'Value':'yield'}, inplace=True)

    if level=='county':
        # Load harvest area 
        corn_area = load_nass_county_data(crop_type, 'areaharvested', 'allstates', 1981, end_year)
        corn_area.rename(columns={'Value':'area'}, inplace=True)
        corn_area.dropna(inplace=True)
    
        # Combine yield and harvest area
        df_final = corn_yield[['Year','FIPS','County','State','yield']].dropna() \
                                                         .merge(corn_area[['Year','FIPS','area']],
                                                                on=['Year','FIPS'],how='left')
    else:
        df_final = corn_yield[['Year','State','yield']].dropna()

    df_final.rename(columns={'Year':'year'}, inplace=True)
    df_final.to_csv('../../crop_modeling/data/nass_yield_area_1981_%d_%s_%s.csv'%(end_year,crop_type,level), index=False)
    print('NASS data for %s saved to csv file in the crop modeling data folder'%crop_type)

if __name__ == "__main__":
#    save_nass_yield_area()
#    save_nass_yield_area(crop_type='soybean')
#    save_nass_yield_area_FV(crop_type='tomatoes')
#    save_nass_yield_area_FV(crop_type='potatoes')
    save_nass_yield_area_FV(crop_type='sweetcorn',level='state')
