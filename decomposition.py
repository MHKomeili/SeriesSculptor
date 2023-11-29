from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


# used for the file dialog panel.
root = tk.Tk()
root.withdraw()


def decomposition():
    
    # ask for input file
   file = fd.askopenfile()
   file_name = file.name    

   # change the dir name so it work on files out of the exe folder path
   if getattr(sys, 'frozen', False):

   # if you are running in a |PyInstaller| bundle
      extDataDir = sys._MEIPASS
      extDataDir  = os.path.join(extDataDir, file_name) 

   #you should use extDataDir as the path to your file Store_Codes.csv file
   else:

   # we are running in a normal Python environment
      extDataDir = os.getcwd()
      extDataDir = os.path.join(extDataDir, file_name) 

   #you should use extDataDir as the path to your file Store_Codes.csv file
   # Prepare our data for decomposition part
   df = pd.read_csv(extDataDir)
   df = df[['Date', 'Total']]
   df['Date'] = pd.to_datetime(df['Date'])
   df.set_index('Date', inplace=True)
   df['Total'] = pd.to_numeric(df['Total'])
   df['log_value'] = np.log(df['Total'])
   
   # Decompose to Trend / Seasonal / Residual
   decomposition_result = seasonal_decompose(df['log_value'], model='additive')

   # Concat Trend / Seasonal / Residual into one dataframe
   output_df = pd.concat([decomposition_result.trend,
                           decomposition_result.seasonal,
                           decomposition_result.resid],axis=1)
   
   # Save the result of decomposition
   output_df.to_csv('decomposition_output.csv')


   ## Time Series Forcasting Part
   # SARIMAX
   df.index = pd.DatetimeIndex(df.index.values,
                               freq=df.index.inferred_freq)
   sarimax_mod = sm.tsa.statespace.SARIMAX(df['log_value'],
                                             trend='c',
                                             order=(1,0,2),
                                             seasonal_order=(0,1,1,12),
                                             enforce_invertibility=False,
                                             enforce_stationarity=False)
   sarimax_results = sarimax_mod.fit(maxiter=3000, full_output=False)
   # # print(sarimax_results.summary())
   # # start_dt = dt.datetime(2023,1,1)
   graph_dt = df.index[0]
   sarimax_df = df[['log_value']].copy()
   sarimax_df['one_step_forecast'] = sarimax_results.predict(dynamic=False)
   # sarimax_df['dynamic_forecast'] = sarimax_results.predict(start=start_dt, dynamic=True)

   sarimax_df[['log_value', 'one_step_forecast']][sarimax_df.index >= graph_dt].plot(figsize=(8, 6), title='SARIMAX One Step Forecast')

   sarimax_forecast = sarimax_results.get_forecast(steps=40)
   sarimax_ci = sarimax_forecast.conf_int()

   sarimax_ci.to_csv('sarimax_forecast.csv')

   ax = df['log_value'][sarimax_df.index >= graph_dt].plot(label='observed', figsize=(10,7))
   sarimax_forecast.predicted_mean.plot(ax=ax, label='Forecast')
   ax.fill_between(sarimax_ci.index,
                sarimax_ci.iloc[:, 0],
                sarimax_ci.iloc[:, 1], color='k', alpha=.25)
   plt.title('SARIMAX Forecast')
   plt.savefig('SARIMAX Forecast.png')

decomposition()

