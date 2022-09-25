# Time-Series-2022-Projections
This repository includes Python codes for the ARIMA predictions of 2022 commodity prices
***

Source: Author’s calculations based on World Bank data and Python code adjusted from 
https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

______________________________________________________________________________________________________________

2022 main commodity prices forecast: Price data for the commodities selected during step 1 are taken 
from the World Bank dataset on monthly commodity prices from January 1970 to April 2022. Out-of-sample 
forecasts were calculated for April to December 2022 using the ARIMA (p,d,q) model, which replicates 
the observed evolution pattern of the historical price changes to the forecast period. 

Estimation of the change in the value of export commodities: The 2022 average price increase for each
commodity is calculated using the commodity price forecasts for April to December 2022 obtained in step 2. 
2022 export values for all Central Asia countries are calculated by multiplying the vector of the estimated 
average price changes between 2021 and 2022 with the export value of the given commodities in 2021, assuming 
that the value of the rest of the exports does not change compared to 2021, and that the 2021 export volume of 
the main commodities stays constant in 2022. 

______________________________________________________________________________________________________________

The code preduces an Excel file in which the monthly price estimates are appended to the end of the rows.
To be able to run the code set the directory to the folder where you download the excel file "Commodity Prices.xlsx" 
which is in this repository.
