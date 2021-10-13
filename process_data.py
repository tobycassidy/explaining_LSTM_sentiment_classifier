import pandas as pd
import tensorflow as tf
import nlp_utils


df_train = pd.read_csv("./assets/data/Corona_NLP_train.csv")
df_test = pd.read_csv("./assets/data/Corona_NLP_test.csv")

df_train["TweetAt"] = [
    date[-4:] + "-" + date[3:5] + "-" + date[0:2] for date in df_train["TweetAt"]
]
df_test["TweetAt"] = [
    date[-4:] + "-" + date[3:5] + "-" + date[0:2] for date in df_test["TweetAt"]
]

df_train["sentiment_score"] = df_train["Sentiment"].replace(
    {
        "Extremely Negative": 1,
        "Negative": 2,
        "Neutral": 3,
        "Positive": 4,
        "Extremely Positive": 5,
    }
)

df_test["sentiment_score"] = df_test["Sentiment"].replace(
    {
        "Extremely Negative": 1,
        "Negative": 2,
        "Neutral": 3,
        "Positive": 4,
        "Extremely Positive": 5,
    }
)

# process text, this is a bottleneck so de-coupled from the app.py and stored processed data in separate files to save time on loading
df_train["clean_tweet"] = df_train["OriginalTweet"].apply(
    lambda x: nlp_utils.clean_text(x)
)
df_test["clean_tweet"] = df_test["OriginalTweet"].apply(
    lambda x: nlp_utils.clean_text(x)
)

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.to_csv("./assets/data/processed_train.csv")
df_test.to_csv("./assets/data/processed_test.csv")
