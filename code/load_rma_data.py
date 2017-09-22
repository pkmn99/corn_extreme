import numpy as np
import pandas as pd
from load_nass_county_data import load_nass_county_data

# Delect extra space in a string
def del_space(s):
    return s.strip()

"""
Load RMA data for a single year (1989-2016)
d = load_rma_data(year)
"""
def load_rma_loss(year):
    if 1989<=year<=2000:
        filename = '../data/rma/COLMNT' + str(year)[-2::] + '.TXT'
        # http://www.rma.usda.gov/data/col/month/colindemnitieswithmonth_1989-2000.pdf
        col_names = ['Commodity Year', 'Locations State Code', 'Location State Abbreviation',
                     'Location County Code', 'Location County Name', 'Commodity Code',
                     'Commodity Name', 'Insurance Plan Code', 'Insurance Plan Abbreviation',
                     'Stage Code', 'Damage Cause Code', 'Damage Cause Description',
                     'Month of Loss', 'Month of Loss Abbreviation','Indemnity Amount']
    if year>=2001:
        filename = '../data/rma/colmnt' + str(year)[-2::] + '.txt'
        # http://www.rma.usda.gov/data/col/month/colindemnitieswithmonth_2001-2010.pdf
        col_names = ['Commodity Year', 'Locations State Code', 'Location State Abbreviation',
                     'Location County Code', 'Location County Name', 'Commodity Code',
                     'Commodity Name', 'Insurance Plan Code', 'Insurance Plan Abbreviation',
                     'Stage Code', 'Damage Cause Code', 'Damage Cause Description',
                     'Month of Loss', 'Month of Loss Abbreviation', 'Determined Acres', 
                     'Indemnity Amount']
        
    d = pd.read_csv(filename, sep='|', header=None, names=col_names, index_col=False,
                    dtype={'Locations State Code': str, 'Location County Code': str,
                           'Commodity Code': str, 'Damage Cause Code': str})
    # delete extra space
    d['Commodity Name'] = d['Commodity Name'].apply(del_space)
    d['Damage Cause Description'] = d['Damage Cause Description'].apply(del_space)
    d['FIPS']=d['Locations State Code'] + d['Location County Code']
    return d

"""
Load RMA loss data of all years for a crop type
df = load_rma_loss_all(crop_name='corn')
"""
def load_rma_loss_all(crop_name='corn'):
    frame = [load_rma_loss(i) for i in range(1989,2016)]
    data_loss = pd.concat(frame)
    if crop_name != 'all':
        data_loss = data_loss[(data_loss['Commodity Name']==crop_name.upper())]
    return data_loss

"""
Load RMA SOB data of all years for a crop type
df = load_rma_sob_all(crop_name='corn')
"""
def load_rma_sob_all(crop_name='corn'):
    frame = [load_rma_sob(i) for i in range(1989,2016)]
    data_sob = pd.concat(frame)
    if crop_name != 'all':
        data_sob = data_sob[(data_sob['Commodity Name']==crop_name.upper())]
    return data_sob


"""
Load RMA summary of businuess data of a single year
df = load_rma_sob(year)
"""
def load_rma_sob(year):
    filename = '../data/rma/sobcov' + str(year)[-2::] + '.txt'
    col_names = ['Commodity Year', 'Locations State Code', 'Location State Abbreviation',
                 'Location County Code', 'Location County Name', 'Commodity Code',
                 'Commodity Name', 'Insurance Plan Code', 'Insurance Plan Abbreviation',
                 'Coverage Category', 'Delivery Type', 'Coverage Level',
                 'Policies Sold Count', 'Policies Earning Premium Count', 'Policies Indemnified Count',
                 'Units Earning Premium Count', 'Units Indemnified Count', 'Quantity Type',
                 'Net Reported Quantity', 'Endorsed/Companion Acres', 'Liability Amount',
                 'Total Premium Amount', 'Subsidy Amount', 'Indemnity Amount',
                 'Loss Ratio']

    d = pd.read_csv(filename, sep='|', header=None, names=col_names, index_col=False,
                    dtype={'Locations State Code': str, 'Location County Code': str,
                           'Commodity Code': str})
    d['FIPS']=d['Locations State Code'] + d['Location County Code']

    # delete extra space
    d['Commodity Name'] = d['Commodity Name'].apply(del_space)
    d['Quantity Type'] = d['Quantity Type'].apply(del_space)
    return d

""" 
Read loss ratio from 1989 to 2016 at county or national levels
df = load_rma_loss_ratio(crop_name='all', level='county')
"""
def load_rma_loss_ratio(crop_name='all', level='county'):
    frame = [load_rma_sob(i) for i in range(1989,2016)]
    data = pd.concat(frame)
    if crop_name != 'all':
        data = data[(data['Commodity Name']==crop_name.upper())]

    if level == 'county':
        loss_ratio = data.groupby(['FIPS', 'Commodity Year']).sum()['Indemnity Amount']\
        /data.groupby(['FIPS', 'Commodity Year']).sum()['Total Premium Amount']
    if level == 'national':
        loss_ratio = data.groupby(['Commodity Year']).sum()['Indemnity Amount']\
        /data.groupby(['Commodity Year']).sum()['Total Premium Amount']
   # return loss_ratio.reset_index().rename(columns={'Commodity Year':'Year',0:'Loss_ratio'})
    return loss_ratio.reset_index().rename(columns={0:'Loss_ratio'})


"""
Load loss ratio by damage cause
df = load_rma_loss_ratio_cause(crop_name='cron')
"""
def load_rma_loss_ratio_cause(crop_name='corn'):
    # load RMA loss and SOB data
    data_loss = load_rma_loss_all(crop_name=crop_name)
    data_sob = load_rma_sob_all(crop_name=crop_name)

    # Indemnity from RMA loss data (sum by FIPS, Year)
    data_loss_cause = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description']).sum()['Indemnity Amount']
    data_loss_cause_sum = data_loss.groupby(['FIPS','Commodity Year']).sum()['Indemnity Amount']
    
    # Indemnity from RMA SOB data (sum by FIPS, Year)
    data_sob_sum = data_sob.groupby(['FIPS','Commodity Year']).sum()
    data_loss_sum = data_loss.groupby(['FIPS','Commodity Year']).sum()
    
    # Get solution from https://stackoverflow.com/questions/20383972/binary-operation-broadcasting-across-multiindex
    # Calculate percentage of indemnity amount by cause
    data_loss_cause_percent = data_loss_cause.unstack('Damage Cause Description'). \
        div(data_loss_cause_sum, axis=0).stack('Damage Cause Description'). \
        reorder_levels(data_loss_cause.index.names)
    
    # Loss ratio from RMA SOB data 
    loss_ratio = load_rma_loss_ratio(crop_name='corn', level='county')
    # loss_ratio.rename(columns={'Year':'Commodity Year'}, inplace=True)
    loss_ratio.set_index(['FIPS', 'Commodity Year'], inplace=True)
    
    # Loss ratio disaggregated into different causes
    loss_ratio_cause = data_loss_cause_percent.unstack('Damage Cause Description'). \
        mul(loss_ratio['Loss_ratio'], axis=0).stack('Damage Cause Description')
    # Merge all these variables 
    result = loss_ratio_cause.reset_index(). \
        rename(columns={0:'Loss ratio by cause'}). \
        merge(loss_ratio.reset_index().rename(columns={'Loss_ratio':'Loss ratio all cause'})).\
        merge(data_loss_cause_percent.to_frame().reset_index(). \
            rename(columns={0:'Cause percent'})). \
        merge(data_loss_cause.reset_index(). \
          rename(columns={'Indemnity Amount':'Indemnity Amount by cause'})). \
        merge(data_sob_sum['Indemnity Amount'].to_frame().reset_index(). \
          rename(columns={'Indemnity Amount':'Indemnity Amount sum SOB'})). \
        merge(data_loss_cause_sum.to_frame().reset_index(). \
          rename(columns={'Indemnity Amount':'Indemnity Amount sum loss'}))
    return result.set_index(['FIPS', 'Commodity Year', 'Damage Cause Description'])


""" 
County loss ratio read all data from 1989 to 2016
Only for corn
df = get_rma_acer_coverage(crop_name='corn', level='county')
"""
def get_rma_acres_coverage(crop_name='corn', level='county'):
    frame = [load_rma_sob(i) for i in range(1989,2016)]
    data = pd.concat(frame)
    if crop_name != 'all':
        data = data[(data['Commodity Name']==crop_name.upper())&(data['Quantity Type']=='Acres')]
    if level == 'county':
        insured_acres = data.groupby(['FIPS', 'Commodity Year']).sum()['Net Reported Quantity']
        insured_acres = insured_acres.reset_index().rename(columns={'Commodity Year':'Year'})
    
    corn_area = load_nass_county_data('corn', 'grain_areaharvested', 'allstates', 1989, 2016)
    df = pd.merge(corn_area, insured_acres, on=['FIPS','Year'])
    df['Coverage rate']=(df['Net Reported Quantity']/df['Value']).clip(0,1) # make the value 0-1
    return df[['FIPS','Year','Coverage rate']]        


"""
Get RMA loss ration by cause and by month
loss_ratio_cause_month = load_rma_loss_ratio_cause_month(rerun=False)
""" 
def load_rma_loss_ratio_cause_month(rerun=False):
    if rerun:
        data_loss = load_rma_loss_all()
        data_sob = load_rma_sob_all()

        # Indemnity from RMA loss data (sum by FIPS, Year)
        data_loss_cause = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description',
                                             'Month of Loss Abbreviation']).sum()['Indemnity Amount']
        data_loss_cause_sum = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description'])\
            .sum()['Indemnity Amount']

        # Indemnity from RMA SOB data (sum by FIPS, Year)
        data_sob_sum = data_sob.groupby(['FIPS','Commodity Year']).sum()
        data_loss_sum = data_loss.groupby(['FIPS','Commodity Year']).sum()

        # Get solution from https://stackoverflow.com/questions/20383972/binary-operation-broadcasting-across-multiindex
        # Calculate percentage of indemnity loss by cause
        data_loss_cause_percent = data_loss_cause.unstack('Month of Loss Abbreviation'). \
            div(data_loss_cause_sum, axis=0).stack('Month of Loss Abbreviation'). \
            reorder_levels(data_loss_cause.index.names)
        
        # Based on loss_ratio_cause to get one level deeper to month
        loss_ratio_cause = load_rma_loss_ratio_cause()

        # Loss ratio disaggregated into different causes and different months
        loss_ratio_cause_month = data_loss_cause_percent.unstack('Month of Loss Abbreviation'). \
            mul(loss_ratio_cause['Loss ratio by cause'], axis=0).stack('Month of Loss Abbreviation')
        loss_ratio_cause_month.to_csv('../data/result/RMA_loss_ratio_cause_month.csv')
        print('Rerun function, file RMA_loss_ratio_cause_month.csv saved')
    else:    
        loss_ratio_cause_month = pd.read_csv('../data/result/RMA_loss_ratio_cause_month.csv')
        print('Do not rerun function, load data from file RMA_loss_ratio_cause_month.csv')
    
    return loss_ratio_cause_month 
