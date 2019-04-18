#Merge the raw datasets and make a proper dataset for preprocessing

import pandas as pd
import numpy as np
import csv
import seaborn as sns

# Load the weather data and take a look at it
print('Weather data looks like :')
print('------------------------\n\n')
wdr = pd.read_csv('weatherDelhi.csv')
print(wdr.head())
print(wdr.columns)
print(wdr.dtypes)

# Load the weather data and take a look at it
print('\n\nAQI data looks like : ')
print('-----------------------\n')
pm = pd.read_csv('pmDelhi.csv')
print(pm.head())
print(pm.columns)
print(pm.dtypes)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16,9]

#First we need to combine the weather and AQI data, and then process it, then remove absolutely unnecessary features as may seem to us

#we will keep the PM data of multiples of 3 for which weather is available, rest we ll delete

# with open('pmDelhi.csv', 'rb') as csvfile: #get hours from the database of pm, int64 type
#     content = csv.reader(csvfile, delimiter=',')
#     for row in content:
#         print(row[6])

frstRow = 1 
with open('pmDelhi.csv', 'rb') as inp, open('pmDelhi2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
    	if frstRow==1: #the first row
    		writer.writerow(row)
    		frstRow=0
    	else:
	    	chkVal=row[6]
	    	# print(str(chkVal))
	    	if (str(chkVal)=='00' or str(chkVal)=='03' or str(chkVal)=='06' or str(chkVal)=='09' or str(chkVal)=='12' or str(chkVal)=='15' or str(chkVal)=='18' or str(chkVal)=='21'):
	    		writer.writerow(row)

#weather data of 2016 and 2017 contains every half an hour, remove that

frstRow = 1 
with open('weatherDelhi.csv', 'rb') as inp, open('weatherDelhi2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
    	if frstRow==1: #the first row
    		writer.writerow(row)
    		frstRow=0
    	else:
    		mytime=row[0]
    		# print(mytime.hour)
    		# print(type(mytime)) #string
    		val=mytime[9:11]
    		# print(val)
	    	if (mytime[12]!='3' and (val=='00' or val=='03' or val=='06' or val=='09' or val=='12' or val=='15' or val=='18' or val=='21')):
	    		writer.writerow(row)

#THERE ARE SOME MISSING ROWS in weather(6763 - 6711), assuming data for every 3 hours intevals
#example 0600 data for 17-2-2015 is missing. We need to fix this first
#Either add duplicate values same as previous row, or ignore corresponding rows for pm2.5, done manually DATA CLEANING AND ARRANGING

#Now we need to add AQI data with weather data (using this bad code)
frstRow = 1 
with open('weatherDelhi3.csv', 'rb') as inp, open('weatherAndAQIdelhi.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if frstRow==1:
            row.append('year')
            row.append('month')
            row.append('day')
            row.append('hour')
            row.append('AQI')
            writer.writerow(row)
        else:
            count=1
            # row[1]=str(hashval(row[1]))
            with  open('pmDelhi3.csv', 'rb') as inp2:
                for row2 in csv.reader(inp2):
                    if count==frstRow:
                        val1=str(row2[3])
                        val2=str(row2[4])
                        val3=str(row2[5])
                        val4=str(row2[6])
                        val5=str(row2[8])
                        row.append(val1)
                        row.append(val2)
                        row.append(val3)
                        row.append(val4)
                        if val5=='-999':
                            val5='0'
                            # break   #DONT WRITE THE ROW IF VAL = -999
                        row.append(val5)
                        writer.writerow(row)
                        break
                    else:
                        count+=1
        frstRow+=1

#this works, now data ready