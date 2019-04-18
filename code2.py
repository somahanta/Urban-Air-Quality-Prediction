#Data preprocessing and cleaning

import pandas as pd
import numpy as np
import csv
import seaborn as sns
from datetime import datetime

condStr = {}
condNum = {}

def hashval(mystr):
    val=0
    # for c in mystr:
    #     val=val+(ord)c
        # print(ord(c))
    val=sum(bytearray(mystr))
    return val


# Load the weather and AQI data and take a look at it
print('Weather and AQI data of Delhi looks like :')
print('------------------------\n\n')
df = pd.read_csv('weatherAndAQIdelhi.csv')
print(df.head())
print(df.columns)
print(df.dtypes)

frstRow = 1 
with open('weatherAndAQIdelhi.csv', 'rb') as inp, open('weatherAndAQIdelhi22.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if frstRow==1:
            writer.writerow(row)
            frstRow=frstRow+1
        else:
        	val=hashval(row[1])
        	condStr[row[1]]=1
        	condNum[val]=1
        	row[1]=str(val)
        	writer.writerow(row)
     

df = pd.read_csv('weatherAndAQIdelhi22.csv')
print(df.head())
print(df.columns)
print(df.dtypes)

print('size of conds for string and numbers are: ')
print(len(condStr))
print(len(condNum))

#remove unneeded features, which are absolutely not useful
#Data that is available from several weather API forecasters: Temperature,Humidity,Wind speed,Wind direction,Dew point

#WE WILL DROP UNNECESSARY FEATURES LATER, lets now analyze graphs - done


df = df.iloc[433:]  #because AQI values are not recorded before that 
print('Dropping heads : ')
print(df.head(10))
df.to_csv('weatherAndAQIdelhi_temp.csv')



import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16,9]

#analyze AQI by various factors

x = df['conds']
y = df['AQI']
plt.scatter(x,y)
plt.xticks(rotation=90)
plt.xlabel('conds')
plt.ylabel('AQI val')
plt.title('AQI by conds')
plt.show()

x = df['dewptm']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('dewptm')
plt.ylabel('AQI val')
plt.title('AQI by dewptm')
plt.show()

x = df['fog']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('fog')
plt.ylabel('AQI val')
plt.title('AQI by fog')
plt.show()

x = df['hum']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('hum')
plt.ylabel('AQI val')
plt.title('AQI by hum')
plt.show()

x = df['pressurem']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('pressurem')
plt.ylabel('AQI val')
plt.title('AQI by pressurem')
plt.show()

x = df['wspdm']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('wspdm')
plt.ylabel('AQI val')
plt.title('AQI by wspdm')
plt.show()

x = df['hour']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('Hour')
plt.ylabel('AQI val')
plt.title('AQI by Hour')
plt.show()

x = df['rain']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('wind speed')
plt.ylabel('AQI val')
plt.title('AQI by Windspeed')
plt.show()

x = df['tempm']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('tempm')
plt.ylabel('AQI val')
plt.title('AQI by tempm')
plt.show()

x = df['wdird']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('wdird')
plt.ylabel('AQI val')
plt.title('AQI by wdird')
plt.show()

x = df['month']
y = df['AQI']
plt.scatter(x,y)
plt.xlabel('month')
plt.ylabel('AQI val')
plt.title('AQI by month')
plt.show()
#NEED TO FILL MISSING VALUES

#drop most useless/ missing data columns first
#remove thunder, tornado,windgust, windchillm, winddir, vism, snow, rain, heatindexm, precipitaion, 

#df.drop(['thunder','tornado','wgustm','windchillm','wdire','vism','snow','rain','heatindexm','precipm'], axis=1)

cols = list(pd.read_csv("weatherAndAQIdelhi_temp.csv", nrows =1))
print(cols)
df= pd.read_csv("weatherAndAQIdelhi_temp.csv", usecols =[i for i in cols if (i != 'thunder' and i != 'tornado' and i != 'wgustm' and i != 'windchillm' and
i != 'wdire' and i != 'vism' and i != 'snow' and i != 'rain' and i != 'heatindexm' and i != 'precipm' and i!='hail' and i!='fog' and i!='year')])
print(df.columns)
df = df.drop(df.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 

print(df.head())

df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df = df.rename(columns={"datetime_utc": "date"})

def time_series(start, end):
    time_series_df = df[['date', 'AQI']][(df['date'] >= start) & (df['date'] <= end)]
    x = time_series_df.date
    y = time_series_df.AQI
    plt.plot(x,y)
    plt.xlabel('Time')
    plt.ylabel('AQI Value')
    plt.title('AQI Time Series')
    return plt.show();

# print(df.head())

time_series('2015','2017')

time_series('2016-09-04','2016-09-07')

#Handle NaNs - remove rows or interpolate

print('\nDoes Nan exist?')
print(df.isnull().values.any())

print('\nDescription of the dataset \n')
print(df.describe())

#Handle NaNs for each feature, then prediction

print(df[df.isnull().any(axis=1)].shape)

# 1. Check if date, month, day or hour have NaNs
print("Date contains nulls:", df.date.isnull().values.any())
print("Month contains nulls:", df.month.isnull().values.any())
print("day contains nulls:", df.day.isnull().values.any())
print("Hour contains nulls:", df.hour.isnull().values.any())

print('\n')
print("AQI contains nulls:", df.AQI.isnull().values.any())

#now check for nan for    conds,dewptm,hum,pressurem,tempm,wdird,wspdm

print("Condns contains nulls:", df.conds.isnull().values.any())
print("Dewptm contains nulls:", df.dewptm.isnull().values.any())
print("Hum contains nulls:", df.hum.isnull().values.any())
print("Pressurem contains nulls:", df.pressurem.isnull().values.any())
print("Tempm contains nulls:", df.tempm.isnull().values.any())
print("Wdird contains nulls:", df.wdird.isnull().values.any())
print("Wspdm contains nulls:", df.wspdm.isnull().values.any())

#Results show
# ('Dewptm contains nulls:', True)
# ('Hum contains nulls:', True)
# ('Pressurem contains nulls:', True)
# ('Tempm contains nulls:', True)
# ('Wdird contains nulls:', True)

#Lets see numer of NaNs for each feature

print('\n\n')
print("Shape of dewptm Nans : ",df[df['dewptm'].isnull()].shape)
print("Shape of hum Nans : ",df[df['hum'].isnull()].shape)
print("Shape of pressurem Nans : ",df[df['pressurem'].isnull()].shape)
print("Shape of tempm Nans : ",df[df['tempm'].isnull()].shape)
print("Shape of wdird Nans : ",df[df['wdird'].isnull()].shape)

# ('Shape of dewptm Nans : ', (9, 52))
# ('Shape of hum Nans : ', (29, 52))
# ('Shape of pressurem Nans : ', (40, 52))
# ('Shape of tempm Nans : ', (23, 52))
# ('Shape of wdird Nans : ', (2539, 52))

#for now fill means to all, wdird check later

print('\nFilling NaN values with means or sth\n')
df['dewptm'] = df.dewptm.fillna(df.dewptm.mean())
print("Dewptm contains nulls:", df.dewptm.isnull().values.any())
df['hum'] = df.hum.fillna(df.hum.mean())
print("hum contains nulls:", df.hum.isnull().values.any())
df['pressurem'] = df.pressurem.fillna(df.pressurem.mean())
print("pressurem contains nulls:", df.pressurem.isnull().values.any())
df['tempm'] = df.tempm.fillna(df.tempm.mean())
print("tempm contains nulls:", df.tempm.isnull().values.any())

df['wdird'] = df.wdird.fillna(method='ffill') #fill with preceeding values, no need to have done manually some
print(df[df['wdird'].isnull()]) #the first 2 rows have no previous value to copy hence null
df = df.dropna(axis=0,subset=['wdird'])

print("wdird contains nulls:", df.wdird.isnull().values.any()) 
print("Shape of wdird Nans : ",df[df['wdird'].isnull()].shape)

print('\n\n')
print(df.head())
print('\n\n')

df['conds_1'] = df.conds.shift(periods=1)
df['dewptm_1'] = df.dewptm.shift(periods=1)
df['hum_1'] = df.hum.shift(periods=1)
df['pressurem_1'] = df.pressurem.shift(periods=1)
df['tempm_1'] = df.tempm.shift(periods=1)
df['wdird_1'] = df.wdird.shift(periods=1)
df['wspdm_1'] = df.wspdm.shift(periods=1)
df['AQI_1'] = df.AQI.shift(periods=1)

df['conds_2'] = df.conds.shift(periods=2)
df['dewptm_2'] = df.dewptm.shift(periods=2)
df['hum_2'] = df.hum.shift(periods=2)
df['pressurem_2'] = df.pressurem.shift(periods=2)
df['tempm_2'] = df.tempm.shift(periods=2)
df['wdird_2'] = df.wdird.shift(periods=2)
df['wspdm_2'] = df.wspdm.shift(periods=2)
df['AQI_2'] = df.AQI.shift(periods=2)

df['conds_3'] = df.conds.shift(periods=3)
df['dewptm_3'] = df.dewptm.shift(periods=3)
df['hum_3'] = df.hum.shift(periods=3)
df['pressurem_3'] = df.pressurem.shift(periods=3)
df['tempm_3'] = df.tempm.shift(periods=3)
df['wdird_3'] = df.wdird.shift(periods=3)
df['wspdm_3'] = df.wspdm.shift(periods=3)
df['AQI_3'] = df.AQI.shift(periods=3)

df['conds_4'] = df.conds.shift(periods=4)
df['dewptm_4'] = df.dewptm.shift(periods=4)
df['hum_4'] = df.hum.shift(periods=4)
df['pressurem_4'] = df.pressurem.shift(periods=4)
df['tempm_4'] = df.tempm.shift(periods=4)
df['wdird_4'] = df.wdird.shift(periods=4)
df['wspdm_4'] = df.wspdm.shift(periods=4)
df['AQI_4'] = df.AQI.shift(periods=4)

df['conds_5'] = df.conds.shift(periods=5)
df['dewptm_5'] = df.dewptm.shift(periods=5)
df['hum_5'] = df.hum.shift(periods=5)
df['pressurem_5'] = df.pressurem.shift(periods=5)
df['tempm_5'] = df.tempm.shift(periods=5)
df['wdird_5'] = df.wdird.shift(periods=5)
df['wspdm_5'] = df.wspdm.shift(periods=5)
df['AQI_5'] = df.AQI.shift(periods=5)

print(df.head(10))

#remove first 5 rows which contain NaN
df=df.iloc[5:]
print('\n\n')
print(df.head(10))

print('\nDoes Nan exist anymore anywhere?')
print(df.isnull().values.any())

print('\nFinal data description: ')
print(df.describe())

df.to_csv('weatherAndAQIdelhi_cleaned.csv')
