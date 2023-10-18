import cnbc
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import pyfiglet
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer 


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
        blob = TextBlob(row['title'])
        sentiment_score = blob.sentiment.polarity
        sentiment_scores = sia.polarity_scores(row['title'])
        polarity_label = encode_polarity2(sentiment_scores['compound'])
        print_clickable_link(f">> {row['title']}\n", row['url'])
        print(f"Sentiment Score: {sentiment_scores['compound']}")
        print(f"Polarity Label: {polarity_label} \n")
        
     print(pyfiglet.figlet_format("Negative"))
     for index, row in negative_news.iterrows():
        blob = TextBlob(row['title'])
        sentiment_score = blob.sentiment.polarity
        sentiment_scores = sia.polarity_scores(row['title'])
        polarity_label = encode_polarity2(sentiment_scores['compound'])
        print_clickable_link(f">> {row['title']}\n", row['url'])
        print(f"Sentiment Score: {sentiment_scores['compound']}")
        print(f"Polarity Label: {polarity_label} \n")

# Function to encode polarity labels based on scores
def encode_polarity(score):
    if score > 0:
        return 'Positive news'
    elif score < 0:
        return 'Negative news'
    else:
        return 'Neutral news'


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

news_sentiment()
