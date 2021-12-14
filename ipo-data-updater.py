#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import yfinance as yf
import investpy
import numpy as np


# In[ ]:


df_main = pd.read_excel(r'RawData.xlsx')

### Gather data from IPO
page="http://www.aastocks.com/en/stocks/market/ipo/listedipo.aspx?s=3&o=0&page=" + str (1)

dfs = pd.read_html(page)
df = dfs [16]
df = df [:-3]
df = df.iloc [:,1:]
df2 = df ['Name▼ / Code▼']
df2 = df2.map(lambda x: x.rstrip('Sink Below Listing Price'))
df_code = df2.map(lambda x: x[-7:])
df_name =  df2.map(lambda x: x[:-8])

df ['Name▼ / Code▼'] = df_code
df.insert(0, 'Name', df_name)
df = df.rename(columns = {'Name▼ / Code▼':'Code'})
df_IPO = df[~df['Code'].isin(df_main['Code'])]

### Gather sponsor data
page= 'http://www.aastocks.com/en/stocks/market/ipo/sponsor.aspx?s=1&o=0&s2=0&o2=0&f1=&f2=&page=' + str(1) + '#sponsor'
dfs = pd.read_html(page)
df = dfs [17]
df = df.iloc[:-2,0:7]
df ['Name▼ / Code▼'] = df_code
df.insert(0, 'Name', df_name)
df = df.rename(columns = {'Name▼ / Code▼':'Code'})
df_sponsor = df[df['Code'].isin(df_IPO['Code'])]
df_sponsor = df_sponsor.drop(columns = ['Name/Code', 'List Date', 'Acc. % Chg.▼', '% Chg. onDebut1▼', 'Name' ],axis = 1)

### merge newly gathered data
df_new = df_IPO.merge(df_sponsor, on = ['Code'], how = 'left')

df_new = df_new.rename( columns={'Industry':'AA Stocks Industry'})

def cleanpercent (df_main, column):
    df_main[column]= df_main[column].astype (str)
    df_main[column]= df_main[column].str.replace('%', '', regex=True)
    df_main[column]= df_main[column].str.replace('+', '', regex=True)
    df_main[column]= df_main[column].astype (float)
    df_main[column]= df_main[column]/100
    return df_main

df_new = cleanpercent (df_new, '% Chg. on2Debut▼')
df_new = cleanpercent (df_new, 'Gray Market (%)2')
df_new = cleanpercent (df_new, 'One LotSuccess Rate▼')


# In[ ]:


### Add Yahoo Industries/Sector Data
df2 = df_new ['Code']
df8 = []
for data in df2:  
    try:
        DC = yf.Ticker(data)
        df3 = DC.info  ['industry'] 
        df4 = DC.info ['sector']
        df6 = DC.info ['longBusinessSummary']
        df5 = [df3, df4, df6, data]
        df8.append (df5)
    except KeyError:
        df5 = ['na', 'na', 'na', 'na']
        df8.append (df5)
        
df8 = pd.DataFrame(df8, columns=['Industry', 'Sector', 'Business Summary', 'Code'])

### merge newly gathered data
df_new = df_new.merge(df8, on = ['Code'], how = 'left')

### get new Leads data
df2 = df_new['Sponsor'].str.replace(', Limited', ' Limited,', regex=True)
df2 = df2.str.split (',', expand = True)

def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

df2 = trim_all_columns(df2)

for col in df2.columns:
    df2 = df2.rename(columns = {col:'Lead '+ str(col + 1)})

### merge newly gathered data
df_new = pd.concat([df_new,df2],axis = 1)
        
### Splitting Market Cap and Offer price to 2 columns
def split_name(df, df_column):
    """
    Splits columns with a - in the data and returns a column with the lower and upper bound
    """
    df2 = df [df_column]
    df2 = df2.str.split ('-', expand = True)
    df2 = df2.astype(float)
    df2 = df2.rename(columns = {0:'Lower '+ df_column, 1: 'Upper ' + df_column})
    df [df_column] = df2.iloc [:,0]
    df = df.rename(columns = {df_column:'Lower ' + df_column})
    
    df.insert(df.columns.get_loc('Lower ' + df_column) +1, 'Upper ' + df_column, value = df2.iloc [:,1])
    return df

df_new = split_name(df_new, 'Market Cap(B)')
df_new = split_name(df_new, 'Offer Price')

### calculating Market cap
mktcap = df_new['Listing Price'].astype(float) / df_new['Upper Offer Price'].astype(float) * df_new[ 'Upper Market Cap(B)'].astype(float)
df_new.insert (6, 'Market Cap(B)', value = mktcap)

### cleaning data before merging
df_new = df_new.iloc [::-1]
df_new = df_new.drop(columns = ['Last1','Acc.% Chg.▼' ],axis = 1)

### concat new data with old data
df_main = pd.concat([df_main,df_new], axis=0)


# In[ ]:


## gather yahoo trading data and calculating return
    ### initialize data
df2 = df_main ['Code']
df_date = df_main ['Listing Date▼']
df_date  = pd.to_datetime(df_date, infer_datetime_format=True)
df_date = pd.concat ([df_date, df2], axis=1)
df_date = df_date.set_index('Code')

df = yf.download('^HSI',  start="2018-01-01")  ['Close'] 

today = pd.to_datetime('today').strftime('%d/%m/%Y')
df_invest = investpy.get_index_historical_data(index='hs healthcare',
                                        country='hong kong',
                                        from_date='01/01/2018',
                                        to_date=today)
df_invest = df_invest ['Close']

df_trading8 = []
df_HSI8 = []
df_HSH8 =[]



df2 = df2.values.tolist()
df3 = yf.download(df2, period="max")  ['Close'] 

def calc_data (init_df,date_df,trading_days):
    df_HSI = []
    for day in trading_days:
        try:
            df_day = date_df[day]
            
        except (IndexError, TypeError):
            df_day = 'NAN'     
        df_HSI.append(df_day)
    df_HSI.append (ticker)
    init_df.append(df_HSI)
    return init_df
for ticker in df2:
    df4 = df3 [ticker]
    df4 = df4.dropna()    
    start_date = df_date.loc [ticker] ### use listing date from AA stocks
    start_date = start_date.values[0]
    trading_days = [0, -1, 80, 100, 120, 140, 160, 252, 372] #!!! THIS CAN BE ADJUSTED AS NECESSARY but others need to be adjusted if different from 9 variables!!!!###

    try:
        end_date = df4.index [-1]
            
            ### filtering by date
        df5 = df.loc[start_date:end_date]
        df4 = df4.loc [start_date:end_date]
    
            ### filtering by date for investpy
        end_dateinv = df_invest.index [-1]
        df6 = df_invest.loc[start_date:end_dateinv]
     
    except IndexError:
        df5 = np.NAN
        df4 = np.NAN
        df6 = np.NAN
    calc_data(df_HSI8,df5, trading_days)
    calc_data(df_trading8, df4,trading_days)
    calc_data(df_HSH8, df6,trading_days)

    ###making dataframe then merging
def make_df (ending_term, df_data):
    HSI_days = [str(i)+ ending_term for i in trading_days]
    HSI_days.append('Code')
    df = pd.DataFrame(df_data, columns = HSI_days)
    return df

df9 = make_df (' HSI Days', df_HSI8)
df10 = make_df (' HSH Days', df_HSH8)
df8 = make_df(' Trading Days', df_trading8)

    ### preparing data for division
df_trading = df8.iloc [:,:-1]
df_HSI = df9.iloc  [:,:-1]
df_HSH = df10.iloc  [:,:-1]

df_trading = df_trading.astype(float)
df_HSI = df_HSI.astype(float)
df_HSH = df_HSH.astype(float)

df_listprice = df_main ['Listing Price']
df_listprice = df_listprice.reset_index(drop=True)
df_listprice = df_listprice.astype(float)

df_code = df8 ['Code']

    #Trading return: dividing returns by list price. Need to review some Matrix Division hahaha
def ret (numerator,denominator,code):
    """ 
    Divides two dataframes by transposing and using linear algebra rules
    """
    df_tradingret = numerator.T / denominator -1
    df_tradingret = df_tradingret.T
    df_tradingret = pd.concat ([df_tradingret, code], axis=1)
    return df_tradingret

df_tradingret = ret(df_trading, df_listprice, df_code)

    # Index return: divide HSI by rest of column
df_HSIPO = df_HSI.iloc [:,0]
df_HSI = df_HSI.iloc[:,1:]

df_HSHPO = df_HSH.iloc [:,0]
df_HSH = df_HSH.iloc[:,1:]

df_HSHret = ret(df_HSH,df_HSHPO,df_code)
df_HSIret = ret(df_HSI,df_HSIPO,df_code)

    ### merge Trading and HSI and HSH return
df_yfret = df_tradingret.merge(df_HSIret, on = ['Code'], how = 'left')
df_yfret = df_yfret.merge(df_HSHret, on = ['Code'], how = 'left')

### removing old data then adding new data
df_end = df_main.iloc [:,-3:]
df_end = df_end.reset_index(drop=True)

df_main = df_main.drop(df_main.columns[26:], axis = 1)

df_main = df_main.merge(df_yfret, on = ['Code'], how = 'left')
df_main = pd.concat ([df_main, df_end], axis=1)


# In[ ]:


### cleaning df_main

df_main ['Listing Price'] = df_main ['Listing Price'].astype (str)
df_main ['Listing Price'] = df_main ['Listing Price'].astype (float)

df_main ['Applied lotsfor 1 lot▼'] = df_main ['Applied lotsfor 1 lot▼'].astype (str)
df_main ['Applied lotsfor 1 lot▼'] = df_main ['Applied lotsfor 1 lot▼'].str.replace(' lot', '', regex=True)
df_main ['Applied lotsfor 1 lot▼'] = df_main ['Applied lotsfor 1 lot▼'].astype (float)

df_main ['Over-sub.Rate▼'] = df_main ['Over-sub.Rate▼'].astype (str)
df_main ['Over-sub.Rate▼'] = df_main ['Over-sub.Rate▼'].str.replace ('Under-Sub.', '-1000', regex=True)
df_main ['Over-sub.Rate▼'] = df_main ['Over-sub.Rate▼'].astype (float)

df_main ['Lot Size'] = df_main ['Lot Size'].astype (str)
df_main ['Lot Size'] = df_main ['Lot Size'].astype (float)

df_main ['Listing Date▼'] = df_main ['Listing Date▼'].astype (str)
df_main ['Listing Date▼'] = df_main ['Listing Date▼'].str.replace (' 00:00:00','', regex = True)

    ## removing discrepancy and calculating new discrepancy

discrepancy = abs(df_main ['% Chg. on2Debut▼'] - df_main['0 Trading Days'])
discrepancy = discrepancy.round(2)
df_main ['Discrepancy'] = discrepancy

df_main.to_excel('RawData 2.xlsx', index = False)

