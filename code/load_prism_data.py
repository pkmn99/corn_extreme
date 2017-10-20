"""
Load daily PRSIM county data for a single year, 1981-2015
df = load_prism_county_data('tdeman', 2000, freq='1d')
freq: '1M', '1A'
"""
import numpy as np
from scipy import stats
import pandas as pd

def load_prism_county_year(variable, year, freq='1d'):
    fn_path = '../../data/PRISM/data/county_level/'
    fn = variable + '_daily_' + str(year) +'_county.csv'
    df = pd.read_csv(fn_path + fn, index_col=[0], parse_dates=[0])
#     df.set_index('Unnamed: 0', inplace=True)
    if freq!='1d':
        if variable != 'ppt': # return sum for ppt
            return df.resample(freq).mean()
        else:
            return df.resample(freq).sum()
    else:
        return df

# Load data for a year range    
def load_prism_county_year_range(variable, start_year, end_year, freq='1d'):
    df = [load_prism_county_year(variable, i, freq) for i in range(start_year,end_year+1)]
    df_all = pd.concat(df)
    return df_all.dropna(axis='columns', how='all')

"""
Calculate percentile based on ranking 
Usage: rank = calculate_rank(df, month_range)
month_range = [4,7] Apr to July
"""    
def calculate_rank(df, month_range):
    prng = pd.period_range('1981', '2015', freq='A')
    df_out = pd.DataFrame(data=np.zeros([35,df.columns.shape[0]]), index=prng, 
                          columns=df.columns)
    
    for f in df.columns:
        rank_temp = np.zeros([35,month_range[1]-month_range[0]+1])# temp array to save monthly rank
        for m in range(month_range[0], month_range[1]+1):
            x = df[f][pd.date_range('1981-%02d'%m, periods=35, freq='12M')]
            rank_temp[:,m-month_range[0]] = stats.rankdata(x, "average")/len(x)
        df_out[f] = rank_temp.mean(axis=1)
    return df_out

# """ 
# Percentage of rainfall binned up to 3.5 SD (mean + np.arange(0,4.1,0.5)) from May to August (adjustable).
# Return percentage of daily rainfall that falls into the range 
# df_heavy_rain, df_hrain_time = heavy_rain_percent(months_start=5, month_end=8)
# """
def heavy_rain(percent=False,month_start=5, month_end=8):
    
    prec_daily = load_prism_county_year_range('ppt', 1981, 2015, freq='1D')
    
    # prec_bin_rank = np.arange(0.1,1,0.1) 

    # a big np array to save results
    array = np.zeros([prec_daily.columns.shape[0]*(2015-1981+1),9])
    
    # to save the month when the most extreme rainfall occurs, FIPS x Year x Months (e.g., May to Aug) 
    array2 = np.zeros([prec_daily.columns.shape[0]*(2015-1981+1), month_end-month_start+1])


    k=0
    for fips in prec_daily.columns:

        temp = prec_daily[fips]
     
        # calculate std and mean
        v_mean = temp[(temp.index.month>=month_start)&(temp.index.month<=month_end)]. \
            replace(0, np.nan).dropna().mean()
        v_std = temp[(temp.index.month>=month_start)&(temp.index.month<=month_end)]. \
            replace(0, np.nan).dropna().std()

        prec_bin_sigma = [v_mean + i * v_std for i in np.arange(0,4,0.5)] 
        prec_bin_sigma.insert(0,0)
        prec_bin_sigma.append(10000)

        # growing season data 
        temp2 = temp[(temp.index.month>=month_start)&(temp.index.month<=month_end)].dropna().to_frame('Prec')

        # every from from 1981-2016         
        for y in range(1981,2016):         
            bin_means, bin_edges, binnumber = stats.binned_statistic(temp2.loc[str(y)]['Prec'], 
                                                                    temp2.loc[str(y)]['Prec'], 'sum',
                                                                    bins=prec_bin_sigma)
            if percent:
                array[k,:] = bin_means/np.sum(bin_means)
            else:
                array[k,:] = bin_means
            
#             print fips, y
            # When the most extreme daily rainfall occurs, count  
#             if binnumber==9:
            array2[k,:]=np.histogram(temp2.loc[str(y)]['Prec'].index.month[binnumber>8], 
                            np.arange(month_start,month_end+2)-0.01)[0]
            k = k + 1

    # combine results to dataframe
    if percent:
        heavy_rain_txt = ['HPrec_percent_0', 'HPrec_percent_1','HPrec_percent_2','HPrec_percent_3',
                  'HPrec_percent_4','HPrec_percent_5','HPrec_percent_6','HPrec_percent_7',
                  'HPrec_percent_8']
    else:
        heavy_rain_txt = ['HPrec_0', 'HPrec_1','HPrec_2','HPrec_3',
                  'HPrec_4','HPrec_5','HPrec_6','HPrec_7',
                  'HPrec_8']
        
    hrain_time_txt = [str(e) for e in range(month_start, month_end+1)] 
        
    iterables = [prec_daily.columns.tolist(), range(1981,2016)]
    fips_year_index = pd.MultiIndex.from_product(iterables, names=['FIPS', 'Year'])

    df_heavy_rain = pd.DataFrame(array, index=fips_year_index, columns=heavy_rain_txt)
    df_hrain_time = pd.DataFrame(array2, index=fips_year_index, columns=hrain_time_txt)

    return df_heavy_rain, df_hrain_time


# Load growing season climate, May to Aug
def load_gs_climate(var='ppt'):
    dr = pd.date_range('1981','2016',freq='A')
    df = load_prism_county_year_range(var, 1981, 2016, freq='1M')
    
    temp = (df.iloc[range(4,420,12),:].values + df.iloc[range(5,420,12),:].values +
            df.iloc[range(6,420,12),:].values + df.iloc[range(7,420,12),:].values)
    
    if var != 'ppt':
        temp = temp/4
            
    df_gs = pd.DataFrame(temp, index=dr, columns=df.columns)
    
    return df_gs
