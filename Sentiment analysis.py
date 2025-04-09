import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    
    nltk.download('vader_lexicon')
    
    
    sia = SentimentIntensityAnalyzer()
    
    
    sentiment_scores = sia.polarity_scores(text)
    
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, sentiment_scores


if __name__ == "__main__":
    text = input("Enter a sentence for sentiment analysis: ")
    sentiment, scores = analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}")
