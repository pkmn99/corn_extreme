import pandas as pd
import numpy as np
from scipy import stats
from load_rma_data import load_rma_loss_all, load_rma_sob_all, load_rma_loss_ratio, \
     load_rma_loss_ratio_cause, get_rma_acres_coverage
from load_nass_data import load_nass_county_data, add_yield_anomaly
from load_prism_data import load_prism_county_year_range, calculate_rank, heavy_rain
"""
This is the main data processing, to get bin_yield
"""
# Change to include sept climate to see whether results change 01/09/2018

# Load necessary data and variables 

#fitting_type = 'linear'
fitting_type = 'quadratic'

# Load RMA loss ratio data
loss_ratio_cause = load_rma_loss_ratio_cause()
loss_ratio_cause.index.rename([u'FIPS', u'Year', u'Damage Cause Description'], inplace=True)

# Load corn yield data
corn_yield = load_nass_county_data('corn', 'grain_yield', 'allstates', 1981, 2016)
corn_yield.rename(columns={'Value':'Yield'}, inplace=True)

corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1981, 2016)
corn_area.rename(columns={'Value':'Area'}, inplace=True)
corn_combined = corn_yield.dropna().merge(corn_area[['Year','FIPS','Area']].dropna(),
                                          on=['FIPS','Year'], left_index=True)

# Add yield and area anomaly
if fitting_type == 'quadratic':
    yield_sample, trend_para = add_yield_anomaly(corn_combined, rerun=True, fitting_type=fitting_type)
    # yield_sample, trend_para = add_yield_anomaly(corn_combined, rerun=True)
else:
    yield_sample, trend_para = add_yield_anomaly(corn_combined, rerun=False)

# Combine yield with RMA loss
data_rma_yield = pd.merge(loss_ratio_cause.reset_index(), \
                          yield_sample[['FIPS','Year','Yield','Yield_ana','State']]). \
                          set_index(['FIPS','Year','Damage Cause Description'])  
    
    
# Filter the combined data with top ten damage causes 
cause_names = ['Excess Moisture/Precip/Rain','Drought','Heat','Cold Wet Weather',
               'Hail','Wind/Excess Wind','Flood','Wildlife','Insects','Other (Snow-Lightning-Etc.)',
               'Frost','Freeze']
data_rma_yield_top = data_rma_yield[data_rma_yield.index.isin(cause_names,level=2)] 


# Load climate data
prec_monthly = load_prism_county_year_range('ppt', 1981, 2016, freq='1M')
prec_monthly.index.name='Year'

tmax_monthly = load_prism_county_year_range('tmax', 1981, 2016, freq='1M')
tmax_monthly.index.name='Year'


# Get climat, yield, together (bin_yield) 
# Get extreme before merging with yield

def bin_yield_climate_county(yield_data, fips, add_loss_ratio=False):    

    # Add growing season precipitation
    P = pd.concat([prec_monthly[fips][prec_monthly[fips].index[4:-1:12]].to_period('A').rename('Prec_5'),
           prec_monthly[fips][prec_monthly[fips].index[5:-1:12]].to_period('A').rename('Prec_6'),
           prec_monthly[fips][prec_monthly[fips].index[6:-1:12]].to_period('A').rename('Prec_7'),
           prec_monthly[fips][prec_monthly[fips].index[7:-1:12]].to_period('A').rename('Prec_8')],
           axis=1).reset_index()

   # P = pd.concat([prec_monthly[fips][prec_monthly[fips].index[4:-1:12]].to_period('A').rename('Prec_5'),
   #        prec_monthly[fips][prec_monthly[fips].index[5:-1:12]].to_period('A').rename('Prec_6'),
   #        prec_monthly[fips][prec_monthly[fips].index[6:-1:12]].to_period('A').rename('Prec_7')],
   #        axis=1).reset_index()

    P['Year'] = P['Year'].apply(lambda x: x.year)
    P['Prec'] = P.iloc[:,1::].sum(axis=1)
    P['Prec_percentile']=(stats.rankdata(P['Prec'], method='average')*2-1)/(P['Prec'].shape[0]*2)

    # Add growing season tmax
    T = pd.concat([tmax_monthly[fips][tmax_monthly[fips].index[4:-1:12]].to_period('A').rename('Tmax_5'),
           tmax_monthly[fips][tmax_monthly[fips].index[5:-1:12]].to_period('A').rename('Tmax_6'),
           tmax_monthly[fips][tmax_monthly[fips].index[6:-1:12]].to_period('A').rename('Tmax_7'),
           tmax_monthly[fips][tmax_monthly[fips].index[7:-1:12]].to_period('A').rename('Tmax_8')],
           axis=1).reset_index()

   # T = pd.concat([tmax_monthly[fips][tmax_monthly[fips].index[4:-1:12]].to_period('A').rename('Tmax_5'),
   #        tmax_monthly[fips][tmax_monthly[fips].index[5:-1:12]].to_period('A').rename('Tmax_6'),
   #        tmax_monthly[fips][tmax_monthly[fips].index[6:-1:12]].to_period('A').rename('Tmax_7')],
   #        axis=1).reset_index()

    T['Year'] = T['Year'].apply(lambda x: x.year)
    T['Tmax'] = T.iloc[:,1::].mean(axis=1)
    T['Tmax_percentile'] = (stats.rankdata(T['Tmax'], method='average')*2-1)/(T['Tmax'].shape[0]*2)
    
    temp = P.merge(T)
    
    #Use STD from 1981 to 2010?
    c = temp['Year'] <= 2020
    v_mean = temp[c][['Prec','Tmax']].apply(np.mean, axis=0)
    v_std = temp[c][['Prec','Tmax']].apply(np.std, axis=0)

    # bin yield anomaly and loss 

    prec_bin_sigma = [v_mean['Prec'] + i * v_std['Prec'] for i in np.arange(-3.5,3.6,0.5)]

    prec_bin_rank = np.arange(0,1.0001,0.05)

    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(temp['Prec_percentile'], temp['Prec_percentile'], 'mean', bins=prec_bin_rank)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(temp['Prec'], temp['Prec'], 'mean', bins=prec_bin_sigma)

    temp['Prec_rank_bin'] =  binnumber1
    temp['Prec_sigma_bin'] = binnumber2
    temp['Prec_to_sd'] = (temp['Prec'] - v_mean['Prec'])/v_std['Prec']

    tmax_bin_sigma = [v_mean['Tmax'] + i * v_std['Tmax'] for i in np.arange(-3.5,3.6,0.5)] 
    tmax_bin_rank = np.arange(0,1.00001,0.05)
    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(temp['Tmax_percentile'],temp['Tmax_percentile'], 'mean', bins=tmax_bin_rank)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(temp['Tmax'], temp['Tmax'], 'mean', bins=tmax_bin_sigma)

    temp['Tmax_rank_bin'] =  binnumber1
    temp['Tmax_sigma_bin'] =  binnumber2
    temp['Tmax_to_sd'] = (temp['Tmax'] - v_mean['Tmax'])/v_std['Tmax']

    temp = temp.merge(yield_data[yield_data['FIPS']==fips])

    # Add loss ratio (for drought)
    if add_loss_ratio:
        for cause_txt in cause_names[0:4]:
            temp = temp.merge(data_rma_yield_top.loc[fips].xs(cause_txt,level=1)[['Loss ratio by cause', 'Cause percent']]. \
                       rename(columns={'Loss ratio by cause': 'loss_ratio_'+cause_txt,'Cause percent':'loss_percent_'+cause_txt}).reset_index(),
                             how='outer') 

    return temp  


# Get bin_yield witout loss ratio
F1 = yield_sample['FIPS'].unique()
F2 = prec_monthly.columns.values

add_loss_ratio=False

if add_loss_ratio:
    F3 = data_rma_yield_top.index.get_level_values(0).unique().values
    fips_all = np.intersect1d(np.intersect1d(F1, F2), F3)
else: 
    fips_all = np.intersect1d(F1, F2)

frame = [bin_yield_climate_county(yield_sample, fips, add_loss_ratio) for fips in fips_all]

bin_yield = pd.concat(frame)

# Calculate reduction percent relative to the trend term
bin_yield['Yield_ana_to_yield'] = bin_yield['Yield_ana']/(bin_yield['Yield'] - bin_yield['Yield_ana'])
# Production loss
bin_yield['Production_ana'] = bin_yield['Area'] * bin_yield['Yield_ana']

bin_yield['Yield_ana_to_yield_area'] = bin_yield['Area'] * bin_yield['Yield_ana_to_yield']

# Add normal condition anomaly
con_normal = (bin_yield['Prec_sigma_bin']>=6)&(bin_yield['Prec_sigma_bin']<=8)
bin_yield = bin_yield.join(bin_yield[con_normal].groupby('FIPS').mean()['Yield_ana_to_yield'].to_frame('Yield_ana_to_yield_normal'), how='left',on='FIPS')
bin_yield['Yield_ana_to_yield_normal_diff'] = bin_yield['Yield_ana_to_yield'] - bin_yield['Yield_ana_to_yield_normal']

if fitting_type == 'quadratic':
    bin_yield.to_csv('../data/result/bin_yield_%s.csv'%fitting_type)
else:
    bin_yield.to_csv('../data/result/bin_yield_mon567.csv')



# get bin_yield With loss ratio
F1 = yield_sample['FIPS'].unique()
F2 = prec_monthly.columns.values

add_loss_ratio=True

if add_loss_ratio:
    F3 = data_rma_yield_top.index.get_level_values(0).unique().values
    fips_all = np.intersect1d(np.intersect1d(F1, F2), F3)
else: 
    fips_all = np.intersect1d(F1, F2)

frame = [bin_yield_climate_county(yield_sample, fips, add_loss_ratio) for fips in fips_all]

bin_yield_rma = pd.concat(frame)

# Calculate reduction rate relative to the trend term
bin_yield_rma['Yield_ana_to_yield'] = bin_yield_rma['Yield_ana']/(bin_yield_rma['Yield'] - bin_yield_rma['Yield_ana'])
# Production loss
bin_yield_rma['Production_ana'] = bin_yield_rma['Area'] * bin_yield_rma['Yield_ana']

if fitting_type == 'quadratic':
    bin_yield_rma.to_csv('../data/result/bin_yield_rma_%s.csv'%fitting_type)
else:
    bin_yield_rma.to_csv('../data/result/bin_yield_rma_mon567.csv')

print('Process done, bin_yield, and bin_yield_rma saved')
