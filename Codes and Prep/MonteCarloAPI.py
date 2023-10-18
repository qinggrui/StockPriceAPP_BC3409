import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from pandas_datareader import data as wb
from scipy.stats import norm

app = Flask(__name__)

@app.route("/MonteCarlo", methods = ['POST', 'GET']) #This adds a "/MonteCarlo" to the end of the website

def MonteCarlo():
    MCdata = request.get_json()

    #input data: date, ticker1, ticker2, ticker1weight, Days_to_Predict, num_iterations

    start_time = MCdata['date'] # YYYY-MM-DD
    new_date = start_time.split("T")[0]

    ticker1 = MCdata['ticker1']
    ticker2 = MCdata['ticker2']

    days = MCdata['Days_to_Predict']
    iterations = MCdata['num_iterations']


    def import_stock_data(tickers, start = '2021-12-1'): #Predetermined date
        data = pd.DataFrame()
        if len([tickers]) ==1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Close']
            data = pd.DataFrame(data)
        else:
            for t in tickers:
                data[t] = wb.DataReader(t, data_source='yahoo', start = start)['Close']
        return(data)

    data = import_stock_data([ticker1, ticker2], start = new_date) #Stock ticker and start date goes here


# create weight of ticker and dataframe manipulation

    ticker1Weight = float(MCdata['ticker1Weight'])
    ticker2Weight = float(1 - ticker1Weight)

    data['WeightedHoldings'] = data[ticker1] * ticker1Weight + data[ticker2] * ticker2Weight #add new column of weighted holdings
    data.drop(ticker1, inplace=True, axis=1) #drop BG column
    data.drop(ticker2, inplace=True, axis=1) #drop U96 column
    data = data.iloc[1: , :] #delete first row (day before)
    data.drop(index=data.index[-1],axis=0,inplace=True) #delete last row (current day)


    def daily_returns(data, days, iterations):
        ft = drift_calc(data)
        try:
            stv = log_returns(data).std().values
        except:
            stv = log_returns(data).std()
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
        return dr

    def log_returns(data):
        return (np.log(1+data.pct_change()))
    #Example use
    log_return = log_returns(data)

    def drift_calc(data):
        lr = log_returns(data)
        u = lr.mean()
        var = lr.var()
        drift = u-(0.5*var)
        
        try:
            return drift.values
        except:
            return drift

    def probs_find(predicted, higherthan, on = 'value'):
        if on == 'return':
            predicted0 = predicted.iloc[0,0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
            less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
        elif on == 'value':
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [i for i in predList if i >= higherthan]
            less = [i for i in predList if i < higherthan]
        else:
            print("'on' must be either value or return")
        return (len(over)/(len(over)+len(less)))

    def simulate_mc(data, days, iterations, plot=True):
        # Generate daily returns
        returns = daily_returns(data, days, iterations)
        # Create empty matrix
        price_list = np.zeros_like(returns)
        # Put the last actual price in the first row of matrix. 
        price_list[0] = data.iloc[-1]
        # Calculate the price of each day
        for t in range(1,days):
            price_list[t] = price_list[t-1]*returns[t]
        
        # Plot Option
        if plot == True:
            x = pd.DataFrame(price_list).iloc[-1]
            fig, ax = plt.subplots(1,2, figsize=(14,4))
            sns.distplot(x, ax=ax[0])
            sns.distplot(x, hist_kws={'cumulative':True},kde_kws={'cumulative':True},ax=ax[1])
            plt.xlabel("Stock Price")
            plt.show()
        
        #CAPM and Sharpe Ratio
        
        # Printing information about stock
        try:
            [print(nam) for nam in data.columns]
        except:
            print(data.name)
        print(f"Days: {days-1}")
        print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
        print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
        print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list),0, on='return')}")
    
        plt.figure(figsize=(15,6))
        plt.plot(pd.DataFrame(price_list).iloc[:,0:iterations]) #change upper limit of simulation here
        return pd.DataFrame(price_list)

    # days = int(input("number of days: ")) +1 #user to key in days input
    # iterations = int(input("number of simulations: ")) #user to key in simulation input

    #Example use


    simulate_mc(data, days, iterations)



#Inputs should come in json format
if __name__ == '__main__':
    app.run(debug = True) #Default port is 5000, can add port = ? to edit the port to your desired port




