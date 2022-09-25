#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:54:48 2022

@author: cesaretvalehov
"""

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir('/Users/cesaretvalehov/Downloads')

#GDP Analysis

price_df = pd.read_excel("Commodity Prices.xlsx", index_col=0)

##########################################################################################################
#crude oil estimates 

oil_df = price_df["Crude oil, average"]

plt.clf()
plt.plot(oil_df)
oil_df.plot()
pyplot.show()
plot_acf(oil_df)
plot_pacf(oil_df)

#split training and testing data
split_point = len(oil_df) - 7
dataset, validation = oil_df[0:split_point], oil_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = oil_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,0))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history.append(inverted)
	month += 1


##########################################################################################################
#coal Australian estimates

coal_df = price_df["Coal, Australian"]

plt.clf()
plt.plot(coal_df)
pyplot.show()

#split training and testing data
split_point = len(coal_df) - 7
dataset, validation = coal_df[0:split_point], coal_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = coal_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(3,0,2))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_coal = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_coal, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_coal.append(inverted)
	month += 1
    
    
##########################################################################################################
#Natural Gas Europe    


gas_df = price_df["Natural gas, Europe"]

plt.clf()
plt.plot(gas_df)
pyplot.show()

#split training and testing data
split_point = len(gas_df) - 7
dataset, validation = gas_df[0:split_point], gas_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = gas_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(3,0,3))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_gas = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_gas, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_gas.append(inverted)
	month += 1
    
##########################################################################################################
#Wheat US HRW 
    

wheat_df = price_df["Wheat, US HRW"]

plt.clf()
plt.plot(wheat_df)
pyplot.show()

#split training and testing data
split_point = len(wheat_df) - 7
dataset, validation = wheat_df[0:split_point], wheat_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = wheat_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,0))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_wheat = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_wheat, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_wheat.append(inverted)
	month += 1
    
##########################################################################################################
#Aluminum
    

aluminum_df = price_df["Aluminum"]

plt.clf()
plt.plot(aluminum_df)
pyplot.show()

#split training and testing data
split_point = len(aluminum_df) - 7
dataset, validation = aluminum_df[0:split_point], aluminum_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = aluminum_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(1,0,1))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_aluminum = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_aluminum, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_aluminum.append(inverted)
	month += 1    


##########################################################################################################
#Iron Ore
    

ore_df = price_df["Iron ore, cfr spot"]

plt.clf()
plt.plot(ore_df)
pyplot.show()

#split training and testing data
split_point = len(ore_df) - 7
dataset, validation = ore_df[0:split_point], ore_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = ore_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,3))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_ore = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_ore, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_ore.append(inverted)
	month += 1   

##########################################################################################################
#Copper
    

copper_df = price_df["Copper"]

plt.clf()
plt.plot(copper_df)
pyplot.show()

#split training and testing data
split_point = len(copper_df) - 7
dataset, validation = copper_df[0:split_point], copper_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = copper_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(3,0,2))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_copper = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_copper, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_copper.append(inverted)
	month += 1   

##########################################################################################################
#Zinc
    

zinc_df = price_df["Zinc"]

plt.clf()
plt.plot(zinc_df)
pyplot.show()

#split training and testing data
split_point = len(zinc_df) - 7
dataset, validation = zinc_df[0:split_point], zinc_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = zinc_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(3,0,2))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_zinc = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_zinc, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_zinc.append(inverted)
	month += 1   
    
##########################################################################################################
#Gold
    

gold_df = price_df["Gold"]

plt.clf()
plt.plot(gold_df)
pyplot.show()

#split training and testing data
split_point = len(gold_df) - 7
dataset, validation = gold_df[0:split_point], gold_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = gold_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,3))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_gold = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_gold, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_gold.append(inverted)
	month += 1   

##########################################################################################################
#Silver
    

silver_df = price_df["Silver"]

plt.clf()
plt.plot(silver_df)
pyplot.show()

#split training and testing data
split_point = len(silver_df) - 7
dataset, validation = silver_df[0:split_point], silver_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = silver_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,0))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_silver = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_silver, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_silver.append(inverted)
	month += 1   



#export files



df = pd.read_excel("Commodity Prices.xlsx")
date_index = df["Date Index"].to_list()
forecast_dates = ['2022M04', '2022M05','2022M06','2022M07',                              
                               '2022M08', '2022M09','2022M10','2022M11',
                               '2022M12']
for i in forecast_dates:
    date_index.append(i)
    
##########################################################################################################
#Cotton
    

cotton_df = price_df["Cotton"]

plt.clf()
plt.plot(cotton_df)
pyplot.show()

#split training and testing data
split_point = len(cotton_df) - 7
dataset, validation = cotton_df[0:split_point], cotton_df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


    
#develop model
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


#fitting the model + finding the stationary version of the series 

X = cotton_df.values
months_in_year = 1
differenced = difference(X, months_in_year)
#differenced_2x = difference(differenced, months_in_year)
#differenced_3x = difference(differenced_2x, months_in_year)
result = adfuller(differenced)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#d = 3

#finding p and q

plot_pacf(differenced, lags = 12) #for p
plot_acf(differenced) # for q
#p = 2
#q = 1
  
    
# fit model
model = ARIMA(differenced, order=(2,0,0))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

#out of sample forecast 


forecast = model_fit.forecast(steps=9)
# invert the differenced forecast to something usable
history_cotton = [x for x in X]
month = 4
for yhat in forecast:
	inverted = inverse_difference(history_cotton, yhat, months_in_year)
	print('2022, Month %d: %f' % (month, inverted))
	history_cotton.append(inverted)
	month += 1   



#export files



df = pd.read_excel("Commodity Prices.xlsx")
date_index = df["Date Index"].to_list()
forecast_dates = ['2022M04', '2022M05','2022M06','2022M07',                              
                               '2022M08', '2022M09','2022M10','2022M11',
                               '2022M12']
for i in forecast_dates:
    date_index.append(i)    


price_estimates = pd.DataFrame(list(zip(date_index, history, history_coal, history_gas,
                                        history_wheat,  history_aluminum, history_ore,
                                        history_copper,  history_zinc, history_gold,
                                        history_silver, history_cotton)), 
                                 columns = ["Date Index", "Crude oil, average", "Australian coal", "Natural gas, Europe",
                                            "Wheat HRW, US", "Aluminum", "Iron Ore", "Copper", "Zinc", 
                                            "Gold", "Silver", "Cotton"])

price_estimates.to_excel("commodity_price estimates.xlsx")





