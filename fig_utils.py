import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff



df_train = pd.read_csv('./assets/data/processed_train.csv')
df_test  = pd.read_csv('./assets/data/processed_test.csv')
model    = tf.keras.models.load_model('assets/trained_model')
# show train class imbalance
#x = ['Positive', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative']
#y = [11422,  9917,  7713,  6624,  5481]
x = df_train['Sentiment'].value_counts().index.values
y = df_train['Sentiment'].value_counts().values
percentages = np.round(100 * y / np.sum(y), 2)

train_proportions = go.Figure(
    go.Bar(
        x=x, 
        y=y,
        marker=dict(
            color='#1F3F49',
            line=dict(color='white', width=3)
        )
    )
)

annotations = []
for x_pos, y_pos, percent in zip(x, y, percentages):
    annotations.append(
        dict(
            x=x_pos,
            y=y_pos + 0.05 * np.max(y),
            text=f'{percent}' + '%',
            showarrow=False
        )
    )

train_proportions.update_layout(
    yaxis={'title':'Number of Tweets'},
    xaxis={'title':'Sentiment of Tweet'},
    title='Train dataset',
    annotations=annotations
)

# show test class imbalance 
#x = ['Negative', 'Positive', 'Neutral', 'Extremely Positive', 'Extremely Negative']
#y = [1041,  947,  619,  599,  592]
x = df_test['Sentiment'].value_counts().index.values
y = df_test['Sentiment'].value_counts().values
percentages = np.round(100 * y / np.sum(y), 2)

test_proportions = go.Figure(
    go.Bar(
        x=x, 
        y=y,
        marker=dict(
            color='#1F3F49',
            line=dict(color='white', width=3)
        )
    )
)

annotations = []
for x_pos, y_pos, percent in zip(x, y, percentages):
    annotations.append(
        dict(
            x=x_pos,
            y=y_pos + 0.05 * np.max(y),
            text=f'{percent}' + '%',
            showarrow=False
        )
    )

test_proportions.update_layout(
    yaxis={'title':'Number of Tweets'},
    xaxis={'title':'Sentiment of Tweet'},
    title='Test dataset',
    annotations=annotations
)


location_count = {l:c for l, c in 
                  zip(df_train['Location'].value_counts().head(10).index, 
                      df_train['Location'].value_counts().head(10))}
location_df = []
for location in location_count.keys():
    if location=='London':
        location_df.append(df_train.loc[df_train['Location']==location, :].groupby('TweetAt') \
                .agg({'sentiment_score':'mean'}) \
                .reset_index() \
                .rename(columns={'sentiment_score':f'{location} sentiment'}))
    else:
        temp_df = df_train.loc[df_train['Location']==location, :].groupby('TweetAt') \
                    .agg({'sentiment_score':'mean'}) \
                    .reset_index() \
                    .rename(columns={'sentiment_score':f'{location} sentiment'})
        
        location_df[0] = pd.merge(location_df[0], temp_df, 
                                  left_on=['TweetAt'], right_on=['TweetAt'], how='outer') \
        .sort_values('TweetAt')
    
time_fig = go.Figure()

for location, count in location_count.items():
    if location=='London':
        time_fig.add_trace(go.Scatter(x=location_df[0]['TweetAt'], 
                         y=location_df[0][f'{location} sentiment'].interpolate(),
                         mode='lines+markers',
                         marker=dict(color='#B3C100',
                            line=dict(color='DarkSlateGrey', width=1)),
                         name=f'{location}, {count}',
                         opacity=0.8
    ))
    else:
        time_fig.add_trace(go.Scatter(x=location_df[0]['TweetAt'], 
                         y=location_df[0][f'{location} sentiment'].interpolate(),
                         mode='lines+markers',
                         marker=dict(
                            line=dict(color='DarkSlateGrey', width=1)),
                         name=f'{location}, {count}',
                         opacity=0.8,
                         visible='legendonly'
    ))
    
time_fig.update_layout(yaxis={'title':'Average Sentiment of Tweets'},
                  xaxis={'title':'Time'},
                  showlegend=True,
                  legend={'title':'Top 10 Locations by Total Tweet Count'},
                  margin={'l':30, 'r':30, 'b':30, 't':30})
