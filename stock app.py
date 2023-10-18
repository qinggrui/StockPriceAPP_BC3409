import pandas as pd
import numpy as np
import pickle
import datetime
from csv import writer
import string
from datetime import datetime
import os
import sys
import pprint
import google.generativeai as palm
from textblob import TextBlob
import requests
import cnbc
from transformers import pipeline
import matplotlib.pyplot as plt
from scipy import stats
import pyfiglet
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# START OF PROGRAM

def stockApp():  # Starts the program
    user = 0
    while True:
        try:  # Direct to user functions
            user = int(input("\nWelcome to our Stock News and Price Prediction App. To proceed, please select between 1-8:\n1)Predict SVB Price (LSTM Model)\n2)Check Market Sentiment\n3)Check latest Stock news\n4)Predict Bank liquidity and default of Bank\n5)Efficient Frontier Model for portfolio analysis\n6)Check yield Curve model based on current asset and holdings\n7)Stock App chatbot\n8)Check our Tableau\nInput: "))
            if (user == 1):
                svbPrice()
                break
            elif (user == 2):
                marketSentiment()
                break
            elif (user == 3):
                news_sentiment()
                break
            elif (user == 4):
                Liquidity()
                break
            elif (user == 5):
                portfolioAnalysis1()
                break
            elif (user == 6):
                yieldCurve()
                break
            elif (user == 7):
                stockChatbot()
                break
            elif (user == 8):
                tableauOpener()
                break
            else:
               print("Please enter a valid input between 1-8")
        except:
            print("Error. Invalid input. ")
    return None


def yieldCurve():

    EPS_value = float(input("Please enter expected or actual EPS Value of your company: "))
    Efficiency_value = float(input("Please enter expected or actual Efficiency Growth Rate (%) of your company: "))
    NIM_value = float(input("Please enter expected or actual Net Interest Margin (%) of your company: "))

    # EPS data
    data_EPS = [0.005735, 0.011196, -0.055034, -0.007426, 0.037259, 0.016309, 0.027098]
    data_EPS = [value * 100 for value in data_EPS]
    # Create a boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_EPS)
    plt.title("Boxplot of Data")

    # Calculate the percentage
    percentage_EPS = stats.percentileofscore(data_EPS, EPS_value)
    # Plot the percentage line on the boxplot
    plt.axhline(y=EPS_value, color='red', linestyle='--', label=f'{EPS_value}%')
    plt.title("Expected EPS growth rating against US Financial Institutions")
    plt.legend()
    plt.show()
    # Print the calculated percentage
    print(f"The expected EPS growth of {EPS_value}% corresponds to being the top {percentage_EPS:.2f}% of the US banks EPS Growth.")


    # Efficiency Growth Rate
    data_Efficiency = [-0.001328, 0.000215, -0.002220, -0.001266, 0.000496, 0.001285, -0.000102]
    data_Efficiency = [value1 * 100 for value1 in data_Efficiency]
    # Create a boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_Efficiency)
    plt.title("Boxplot of Data")
    # Calculate the percentage
    percentage_Efficiency = stats.percentileofscore(data_Efficiency, Efficiency_value)
    # Plot the percentage line on the boxplot
    plt.axhline(y=Efficiency_value, color='red', linestyle='--', label=f'{Efficiency_value}%')
    plt.title("Expected Efficiency Growth Rate against US Financial Institutions")
    plt.legend()
    plt.show()
    # Print the calculated percentage
    print(f"The Efficiency Growth Rate of {Efficiency_value}% corresponds to being the top {percentage_Efficiency:.2f}% of the US banks Efficiency Growth Rate.")


    # Net Interest Margin
    data_NIM = [-0.1328, 0.0215, -0.222, -0.1266, 0.0496, 0.1285, -0.0102]
    data_NIM = [value1 * 100 for value1 in data_NIM]
    # Create a boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_NIM)
    plt.title("Boxplot of Data")
    # Calculate the percentage
    percentage_NIM = stats.percentileofscore(data_NIM, NIM_value)
    # Plot the percentage line on the boxplot
    plt.axhline(y=NIM_value, color='red', linestyle='--', label=f'{NIM_value}%')
    plt.title("Net Interest Margin against US Financial Institutions")
    plt.legend()
    plt.show()
    # Print the calculated percentage
    print(f"The Net Interest Margin of {NIM_value}% corresponds to being the top {percentage_NIM:.2f}% of the US banks Net Interest Margin Rate.")

    filepath = r"C:\Users\Qing Rui\Desktop\BC3409 AI in Acc and Finance\Project\Codes and Prep" # Change file path where needed
    filename = 'yieldData.pkl' 
    # Construct the full file path
    file_path = os.path.join(filepath, filename)
    from datetime import datetime

    try:
        with open(file_path, 'rb') as file:
            rf = pickle.load(file)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

    # Convert the "Date" column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Calculate "Maturity (Years)"
    reference_date = datetime.today()  # You can set your own reference date
    df['Maturity (Years)'] = (reference_date - df['Date']).dt.days / 365
    df = df.drop('Date', axis=1)
    df = df.drop('20 Year Bond', axis = 1)
    # Reorder the columns with "Maturity (Years)" as the first column
    df = df[['Maturity (Years)'] + [col for col in df if col != 'Maturity (Years)']]

    # Nelson-Siegel model function
    def nelson_siegel(maturities, beta0, beta1, beta2, tau):
        return beta0 + (beta1 * (1 - np.exp(-maturities / tau)) / (maturities / tau)) + (beta2 * ((1 - np.exp(-maturities / tau)) / (maturities / tau) - np.exp(-maturities / tau)))
    
    fitted_params = []

    # Plotting individual fitted yield curves
    for i, yield_column in enumerate(df.columns[1:]):
        plt.figure(figsize=(10, 6))  # Create a new figure for each yield curve
        maturities = np.arange(1, 11, 0.1)  # Extend maturities for a smoother curve
        fitted_yields = nelson_siegel(maturities, *fitted_params[i])
        
        plt.plot(maturities, fitted_yields, label=f'Fitted Curve for {yield_column}')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Yield (%)')
        plt.title(f'Nelson-Siegel Yield Curve Fitting for {yield_column}')
        plt.legend()
        plt.grid(True)
        
        # Add fitted parameters as text annotations in the middle of the graph
        params = fitted_params[i]
        text = f'Beta0: {params[0]:.4f}\nBeta1: {params[1]:.4f}\nBeta2: {params[2]:.4f}\nTau: {params[3]:.4f}'
        plt.text(5, max(fitted_yields) / 2, text, fontsize=12, ha='center', va='center', bbox={'facecolor': 'white', 'alpha': 0.7})

        plt.show()

    return None




def svbPrice(): # svbPrice based on time series analysis
    
    input_date = input("Please enter a future time to predict (YYYY-MM-DD): ")

    loaded_model = tf.keras.models.load_model('gfgModel.h5')

    df = pickle.load(open('df.pkl', 'rb'))

    scaler = MinMaxScaler()
    sequence_length = 10
    input_data = df[df['Date'] <= input_date].tail(sequence_length)['Open'].values
    input_data_normalized = scaler.transform(input_data.reshape(-1, 1)).reshape(1, -1, 1)

    # Predict the stock price for the specified date
    predicted_price_normalized = loaded_model.predict(input_data_normalized)
    predicted_price = scaler.inverse_transform(predicted_price_normalized)[0][0]

    print(f"The predicted price of SVB on {input_date}: {predicted_price:.5f}")

    return None


def Liquidity():

    product = input("GDP Export of a country in USD$ (Please enter a numerical value): ")
    tradeshare = input("Book Value of company (Please enter a numerical value): ")
    CET1Ratio = input("Please enter Capital Adequacy Tier 1 ratio of the bank (Please enter a numerical value): ")
    recession = input("Enter probability of recession (Please enter a numerical value): ")
    liqsup = input("Please enter Quick Ratio of the Bank (Please enter a numerical value): ")
    GDPgr = input("Enter forecasted GDP Growth of the country for the year (Please enter a numerical value): ")

    new_data = pd.DataFrame({
        'product': [float(product)],         #  value for 'product'
        'tradeshare': [float(tradeshare)],   # value for 'tradeshare'
        'policytot': [float(CET1Ratio)],     # value for 'policytot/Capital Tier 1 Adequacy Ratio'
        'recession': [float(recession)],     # value for 'recession'
        'liqsup': [float(liqsup)],           # value for 'liqsup'
        'GDPgr': [float(GDPgr)]              # value for 'GDPgr'
    })

    # Specify the filename
    filepath = r"C:/Users/Qing Rui/Desktop/BC3409 AI in Acc and Finance/Project" # Change file path where needed
    filename = 'bankDefault_randomForest.pkl' 
    # Construct the full file path
    file_path = os.path.join(filepath, filename)
    try:
        with open(file_path, 'rb') as file:
            rf = pickle.load(file)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

    predictions = rf.predict(new_data)
    pred_default = "Bank Default predicted" if predictions == 1 else "No Bank Default predicted"

    print(pred_default)

    return None


def tableauOpener():

    filepath = r"C:/Users/Qing Rui/Desktop/BC3409 AI in Acc and Finance/Project" # Change file path where needed
    filename = 'BC3409 Tableau.twb' 
    # Construct the full file path
    file_path = os.path.join(filepath, filename)

    # Open the workbook file with the default application
    os.system(f'start "" "{file_path}"')

    return None


def stockChatbot():
    # pip install -q google-generativeai
    # pip install google-generativeai

    palm.configure(api_key='AIzaSyCpn0pj1BFl9VSYrWvedK_8lwOyZQO-VkY')

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name
    print(model)
    # Input prompt
    prompt = input("Please enter a stock or company name: ")
    prompt = (' '.join(["Tell me about today news relating to", prompt, 'company']))

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=800,
        )

    print(completion.result)

    return None


def marketSentiment():

    api_key = 'ffe05f6637e145509af0c36d97f28cf5'
    countryInput = input("Please enter the country you want to look at \n(Please provide a 2 letter country code. E.g USA is US, Germany is DE): ")

    url = f'https://newsapi.org/v2/top-headlines?country={countryInput}&category=business&apiKey={api_key}'

    response = requests.get(url)
    data = response.json()

    articles = data['articles']
    df = pd.DataFrame(articles)

    # Function to encode polarity labels based on scores
    def encode_polarity(score):
        if score >= 0.3 and score < 0.6:
            return 'Is Moderately Positive'
        elif score >= 0.6:
            return 'Is Extremely Positive'
        elif score > 0 and score < 0.3:
            return 'Is Slightly Positive'
        elif score < 0:
            return 'Is Negative'
        else:
            return 'Is Neutral'

    # Iterate through all titles and perform sentiment analysis
    for title_text in df['title']:

        result = 0

        blob = TextBlob(title_text)
        sentiment_score = blob.sentiment.polarity
        result += sentiment_score

        polarity_label = encode_polarity(result)

    print(f"The outlook for {countryInput} Country {polarity_label} with a sentiment score of {result:.2f}")

    return None


def generate_sentiment(text, sentiment_analysis):
    # generate sentiment label (POSITIVE/NEGATIVE)
    sentiment =  sentiment_analysis(text)
    return sentiment[0]['label'] 


def get_news_for_symbol(symbol):
    # retrieve news for given company ticker using cnbc news api
    json_resp = cnbc.list_symbol_news(symbol=symbol,
                                        api_key='e021c35ea9msh1a13c2bc4c675f9p153e4fjsnb793263f70bc')
    all_news = json_resp['data']['symbolEntries']['results']
    return all_news


def process_all_news(all_news, sentiment_analysis):
    all_news_data =[]
    count = 1
    
    # retrieve the key info from each news
    for news in all_news:
        news_info = {
            "title": news["title"],
            "url": news["url"],
            "desc": news["description"],
            # generate sentiment label given news description
            "sentiment": generate_sentiment(news["description"],sentiment_analysis)
        }
        print(f"Processing item {count} of {len(all_news)}", end='\r')
        count+=1
        all_news_data.append(news_info)

    # return in pandas df format
    return pd.DataFrame(all_news_data)


def plot_pie_chart(df, symbol):
    sentiment_counts = df['sentiment'].value_counts()

    # Plotting the pie chart
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    plt.title(f"Sentiment Distribution for {symbol} News")
    plt.show()


def print_clickable_link(text, url):
    # generate clickable text
    clickable_text = f'\033]8;;{url}\033\\{text}\033]8;;\033\\'
    print(clickable_text)


def print_news_by_sentiment(df):
     positive_news = df[df['sentiment'] == 'POSITIVE']
     negative_news = df[df['sentiment'] == 'NEGATIVE']
     print("*News by sentiment, Click on title to view news article.*")
     sia = SentimentIntensityAnalyzer()
     # Get sentiment scores

     print(pyfiglet.figlet_format("Positive"))
     for index, row in positive_news.iterrows():
        sentiment_scores = sia.polarity_scores(row['title'])
        polarity_label = encode_polarity2(sentiment_scores['compound'])
        print_clickable_link(f">> {row['title']}\n", row['url'])
        print(f"Sentiment Score: {sentiment_scores['compound']}")
        print(f"Polarity Label: {polarity_label} \n")
        
     print(pyfiglet.figlet_format("Negative"))
     for index, row in negative_news.iterrows():
        sentiment_scores = sia.polarity_scores(row['title'])
        polarity_label = encode_polarity2(sentiment_scores['compound'])
        print_clickable_link(f">> {row['title']}\n", row['url'])
        print(f"Sentiment Score: {sentiment_scores['compound']}")
        print(f"Polarity Label: {polarity_label} \n")



# Function to encode polarity labels based on scores
def encode_polarity2(score):
    if score >= 0.05:
        return 'Positive news'
    elif score <= 0.05:
        return 'Negative news'
    else:
        return 'Neutral news'


def news_sentiment():
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    symbol = input("Enter Company Ticker: ")
    all_news = get_news_for_symbol(symbol)
    all_news_processed = process_all_news(all_news, sentiment_analysis)
    plot_pie_chart(all_news_processed,symbol)
    print_news_by_sentiment(all_news_processed)


def dataPrep(tickers, weights, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Open']
    # Calculate the portfolio price dynamically without knowing column names
    data['Portfolio_Price'] = (data.iloc[:, :].values.dot(weights))

    # Reset the index to make Date a regular column
    data.reset_index(inplace=True)

    # Create a new DataFrame with only 'Date' and 'Portfolio_Price' columns
    new_df = data[['Portfolio_Price']]
    data = new_df
    data = data.iloc[1: , :] #delete first row (day before)
    data.drop(index=data.index[-1],axis=0,inplace=True) #delete last row (current day)

    # data.plot(figsize=(15,6))

    # Drop rows with missing values (NaN)
    data.dropna(inplace=True)
    return data

def simulate_mc(data, days, iterations, plot=True):

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
    #Example use
    drift_calc(data)

    def daily_returns(data, days, iterations):
        ft = drift_calc(data)
        try:
            stv = log_returns(data).std().values
        except:
            stv = log_returns(data).std()
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
        return dr
    
    iterations = 10000 #user to key in simulation input

    #Example use
    daily_returns(data, days, iterations)

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

    #Example use (probability our investment will return at least 20% over the days specified in our prediction
    probs_find(data, 0.2, on = 'return')
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
    
    
    # Printing information about stock
    try:
        [print(nam) for nam in data.columns]
    except:
        print(data.name)
    print(f"Days: {days-1}")
    print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
    print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
    print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list),0, on='return')}")
   
    return None


def get_tickers_and_weights():
    tickers = []
    weights = []

    while True:
        ticker = input("Enter a ticker (or '0' to finish): ")
        
        if ticker == '0':
            break
        
        weight = input("Enter the weight for {}: ".format(ticker))
        
        try:
            weight = float(weight)
            if weight < 0 or weight > 1:
                print("Weight must be between 0 and 1.")
            else:
                tickers.append(ticker)
                weights.append(weight)
        except ValueError:
            print("Invalid weight. Please enter a valid number between 0 and 1.")

    return tickers, weights

def portfolioAnalysis1():
    
    # # Define your portfolio tickers and weights
    # tickers = ['AAPL', 'MSFT', 'GOOGL']
    # weights = [0.4, 0.4, 0.2]

    tickers, weights = get_tickers_and_weights()

    import datetime
    # Define the date for analysis
    start_date = datetime.datetime(2000, 1, 1)
    today = datetime.date.today()
    one_day_ago = today - datetime.timedelta(days=1)
    # Set end_date to one day ago
    end_date = datetime.datetime(one_day_ago.year, one_day_ago.month, one_day_ago.day)
    print(start_date)

    data = dataPrep(tickers, weights, start_date, end_date)

    days = 60
    #Example use
    price_pred = simulate_mc(data, days, iterations = 10000 , plot = False)
    print(price_pred)

    portfolioRebalancer(tickers, weights, start_date, end_date)

    return None


def portfolioRebalancer(tickers, weights, start_date, end_date):
    # Get historical price data for the portfolio
    portfolio_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Calculate daily returns for the portfolio
    returns = portfolio_data.pct_change().dropna()

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the number of simulations
    num_portfolios = 10000

    results = np.zeros((4, num_portfolios))

    for i in range(num_portfolios):
        # Generate random weights
        portfolio_weights = np.random.random(len(tickers))
        portfolio_weights /= sum(portfolio_weights)
        
        # Expected portfolio return
        portfolio_return = np.sum(mean_returns * portfolio_weights)
        
        # Expected portfolio volatility
        portfolio_stddev = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights)))
        
        # Sharpe ratio
        sharpe_ratio = portfolio_return / portfolio_stddev
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = sharpe_ratio
        results[3,i] = portfolio_weights[0]  # Weight of AAPL

    # Create a DataFrame to store results
    results_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe Ratio', 'AAPL Weight'])

    # Find the portfolio with the highest Sharpe ratio (risk-adjusted return)
    max_sharpe_portfolio = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]

    # Print the current portfolio characteristics
    print("Current Portfolio Characteristics:")
    print("Expected Return:", max_sharpe_portfolio['Return'])
    print("Volatility (Risk):", max_sharpe_portfolio['Risk'])
    print("Sharpe Ratio:", max_sharpe_portfolio['Sharpe Ratio'])

    # Define target characteristics (You can adjust these based on your goals)
    target_return = 0.10
    target_risk = 0.20

    # Optimization: Find the optimal asset weights to meet the target return and risk
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_stddev  # Minimize the negative Sharpe ratio

    # Constraint: The sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: Asset weights are between 0 and 1
    bounds = tuple((0, 1) for asset in range(len(tickers)))

    # Initial guess (starting point for optimization)
    initial_weights = [1.0 / len(tickers) for _ in range(len(tickers))]

    # Perform optimization to find the optimal asset weights
    optimized_weights = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints).x

    # Rebalance the portfolio based on optimized weights
    portfolio_data['Rebalanced_Portfolio'] = (portfolio_data.iloc[:, :].values.dot(optimized_weights))

    # Calculate the characteristics of the rebalanced portfolio
    rebalanced_returns = portfolio_data['Rebalanced_Portfolio'].pct_change().dropna()
    rebalanced_mean_return = rebalanced_returns.mean()
    rebalanced_stddev = rebalanced_returns.std()

    # Print the rebalanced portfolio characteristics
    print("\nRebalanced Portfolio Characteristics:")
    print("Current Assets:", tickers)
    print("Current Asset Weights:", initial_weights)
    print("Optimal Asset Weights:", optimized_weights)
    print("Expected Return:", rebalanced_mean_return)
    print("Volatility (Risk):", rebalanced_stddev)

    # Plot the efficient frontier
    plt.scatter(results_df.Risk, results_df.Return, c=results_df['Sharpe Ratio'], cmap='YlGnBu')
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe_portfolio['Risk'], max_sharpe_portfolio['Return'], marker='x', color='r', s=200, label='Max Sharpe Ratio Portfolio')
    plt.scatter(rebalanced_stddev, rebalanced_mean_return, marker='o', color='g', s=200, label='Rebalanced Portfolio')
    plt.legend()
    plt.show()

    return None



stockApp()
