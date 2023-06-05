from flask import Flask, request, render_template_string
from tradingview_ta import TA_Handler, Interval
import pandas as pd
from datetime import datetime
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import date, timedelta

nltk.download('vader_lexicon')

intervals = [Interval.INTERVAL_5_MINUTES, Interval.INTERVAL_15_MINUTES, Interval.INTERVAL_1_HOUR, Interval.INTERVAL_1_DAY]
interval_names = ["5 minutes", "15 minutes", "1 hour", "1 day"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symbol = request.form.get('symbol')
        trading_analysis_df = fetch_trading_analysis(symbol)
        sentiment_analysis_results, sentiment_summary = perform_sentiment_analysis(symbol)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template_string("""
        <h1>Trading Analysis for {{ symbol }}</h1>
        <h2>Current Date and Time: {{ current_time }}</h2>
        <h2></h2>
        {{ trading_analysis|safe }}
        <h2>Sentiment Analysis Summary</h2>
        {{ sentiment_summary|safe }}
        <h2>Sentiment Analysis Detail</h2>
        {{ sentiment_analysis|safe }}
        """, symbol=symbol, current_time=current_time, trading_analysis=trading_analysis_df.to_html(), sentiment_analysis=sentiment_analysis_results.to_html(), sentiment_summary=sentiment_summary.to_html())
    else:
        return render_template_string("""
        <form method="POST">
            Enter index symbol: <input type="text" name="symbol"><br>
            <input type="submit" value="Submit">
        </form>
        """)

def fetch_trading_analysis(symbol):
    df = pd.DataFrame()
    for interval_name, interval in zip(interval_names, intervals):
        ta = None
        for exchange in ["NASDAQ", "NYSE"]:
            try:
                ta = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=interval)
                analysis = ta.get_analysis().summary
                break
            except:
                continue

        if ta is None:
            return "Symbol not found in NASDAQ or NYSE"
        
        analysis['Index'] = symbol
        analysis['Interval'] = interval_name
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        analysis['Closing Price'] = hist['Close'].iloc[-1]
        df = df._append(pd.Series(analysis), ignore_index=True)
    df.set_index(['Index', 'Interval'], inplace=True)
    return df

def perform_sentiment_analysis(company_name):
    now = date.today().strftime('%Y-%m-%d')
    yesterday = (date.today() - timedelta(days = 1)).strftime('%Y-%m-%d')

    url = "https://gnews.io/api/v4/search"

    querystring = {
        "q":company_name,
        "max":"50",
        "in":"title",
        "from":yesterday,
        "to":now,
        "lang":"en",
        "token":"e8a6e57aa09dbbd0657e6c396485f82b",
    }

    response = requests.get(url, params=querystring)

    news_data = response.json()
    try:
        news_list = [article['title'] for article in news_data['articles'] if article['title']]
    except KeyError:
        news_list = []
        print(f"Warning: No 'articles' in news_data. Full data: {news_data}")

    sentiment_df = pd.DataFrame(news_list, columns=['News'])

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_df['News'].apply(lambda x: sia.polarity_scores(x))

    sentiment_df = pd.concat([sentiment_df, sentiment_scores.apply(pd.Series)], axis=1)

    sentiment_summary = None
    if 'compound' in sentiment_df.columns:
        sentiment_df['Sentiment'] = pd.cut(sentiment_df['compound'], bins=3, labels=["Negative", "Neutral", "Positive"])
        sentiment_summary = sentiment_df.groupby('Sentiment').count()
        sentiment_summary = sentiment_summary.drop(columns=['compound', 'neg', 'neu', 'pos'])

    return sentiment_df, sentiment_summary

if __name__ == '__main__':
    app.run(port=8000)

