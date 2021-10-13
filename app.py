#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import tensorflow as tf

import os
import nlp_utils
import fig_utils 
from sklearn.preprocessing import LabelEncoder

import dash
import dash_core_components as dcc
import dash_html_components as html 
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


# ### load processed data

# In[3]:


df_train = pd.read_csv('./assets/data/processed_train.csv')
df_test  = pd.read_csv('./assets/data/processed_test.csv')


# In[4]:


encoder = LabelEncoder()
encoder.fit(df_train['sentiment_score'])
encoded_Y_train = encoder.fit_transform(df_train['sentiment_score'])
encoded_Y_test  = encoder.transform(df_test['sentiment_score'])
# convert integers to dummy variables (i.e. one hot encoded)
y_train = tf.keras.utils.to_categorical(encoded_Y_train)
y_test  = tf.keras.utils.to_categorical(encoded_Y_test)


# In[5]:


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_train['clean_tweet'])

vocab_size = len(tokenizer.word_counts)
print(f'Vocabulary size: {vocab_size} (no. of unique words)')

word2idx = tokenizer.word_index
idx2word = {index:word for word, index in word2idx.items()}


# In[6]:


# optimal max_seqlen ? 
seq_lengths = np.array([len(s.split()) for s in df_train['clean_tweet']])
print([(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])


# In[7]:


max_seqlen = 64

X_train = tokenizer.texts_to_sequences(df_train['clean_tweet'])
X_test  = tokenizer.texts_to_sequences(df_test['clean_tweet'])

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, max_seqlen)
X_test  = tf.keras.preprocessing.sequence.pad_sequences(X_test, max_seqlen)


# ### model - RNN 

# In[8]:


embedding_dim = 64
embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_seqlen, name='embedding')

model = tf.keras.models.Sequential([
    embedding,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(5, activation='softmax') #activation='sigmoid')
])
model.compile(optimizer='adam',
              #loss='binary_crossentropy',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


tf.random.set_seed(123)
np.random.seed(123)

log_dir = './logs'

if not os.path.exists(log_dir):
    print('Creating Directory...')
    os.makedirs(log_dir)
else:
    print('Directory Exists')
    
    
callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                           histogram_freq=1,
                                           write_graph=True,
                                           write_images=True,
                                           update_freq='epoch',
                                          # profile_batch=2,
                                          # embeddings_freq=1
                                          )

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=2,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)


# In[10]:


model.save('assets/trained_model')


# In[9]:


model = tf.keras.models.load_model('assets/trained_model')


# In[10]:


model.summary()


# In[29]:


# !tensorboard --logdir ./logs


# ### Looking into some predictions

# In[11]:


df_temp = df_test.copy()
df_temp['Actual']     = df_test['sentiment_score']
df_temp['Prediction'] = [arr.argmax() + 1 for arr in np.round(model.predict(X_test), 0)]
df_temp = df_temp.groupby('TweetAt').agg({'Actual':'mean', 'Prediction':'mean'})


actual_pred_time = go.Figure()


actual_pred_time.add_trace(
    go.Scatter(
        x=df_temp.index.values, 
        y=df_temp['Actual'],
        mode='lines+markers',
        marker=dict(
            color='#B3C100',
            line=dict(
                color='darkslategrey', width=1
            )
        ),
        name='Actual',
        opacity=0.8
))
 
actual_pred_time.add_trace(
    go.Scatter(
        x=df_temp.index.values, 
        y=df_temp['Prediction'],
        mode='lines+markers',
        marker=dict(
            color='#1F3F49',
            line=dict(
                color='darkslategrey', width=1
            )
        ),
        name='Prediction',
        opacity=0.8
))
    
actual_pred_time.update_layout(
    yaxis={'title':'Average Sentiment of Tweets'},
    xaxis={'title':'Time'},
    showlegend=True,
    title='Model Predictions on Test Data as a Function of Time',
    margin={'l':30, 'r':30, 'b':30, 't':30})

#actual_pred_time.show()


# ### app

# In[12]:


import base64


# In[14]:


app = dash.Dash(__name__)


tabs_styles = {
    'height': '50px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '20px',
    'fontWeight': 'bold'
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#6AB187',
    'color': 'black',
    'padding': '20px'
}

vertical_arrow   = './assets/images/vertical_arrow.png'
horizontal_arrow = './assets/images/horizontal_arrow.png'
encoded_vertical_arrow  = base64.b64encode(open(vertical_arrow, 'rb').read())
encoded_horizontal_arrow  = base64.b64encode(open(horizontal_arrow, 'rb').read())


app.layout = html.Div([
    dcc.Markdown('''
    ## Explaining an LSTM Sentiment Classifier: COVID-19 Twitter Dataset
    ---
    ''', style={'border-left':'7px #1F3F49 solid', 'padding':'2px 0px 0px 20px', 'width':'98%'}
    ),
    dcc.Tabs([
        dcc.Tab(label='Overview', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            > ## Model Overview
            > This dashboard aims to make the inner workings of an LSTM tensorflow model 
            > transparent and easily accessible. The model is trained to classify tweets 
            > related to covid, based on their sentiment. This overview tab mainly focuses
            > on the underlying data that the classification model is built on, see subsequent tabs
            > for information related to preprocessing and modelling.
            
            > The sentiment of a tweet is categorised as one of the following:
            > - Extremely Positive. (5)
            > - Positive. (4)
            > - Neutral. (3)
            > - Negative. (2)
            > - Extremely Negative. (1)
            
            > For access to the dataset see [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv).
            ''', style={'background-color':'#CED2CC', 'color':'black', 
                        'padding':'2px 20px 1px 20px', 'width':'80%', 
                        'marginBottom':5, 'marginTop':10, 'marginLeft':80,
                        'box-shadow': '2px 2px 2px 2px grey', 'border-radius': '15px', 'display': 'inline-block'}
            ),
            dcc.Markdown('''
            ### Sentiment Proportions on Train and Test 
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.Div([
            dcc.Graph(figure=fig_utils.train_proportions, style={'width':'45%', 'margin-bottom':'24px', 'margin-left': '10px',
                                                       'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)', 'border-radius': '15px', 'display': 'inline-block',
                                                       'background-color':'#f9f9f9', 'padding': '3px'
                                                       }),
            dcc.Graph(figure=fig_utils.test_proportions, style={'width':'45%', 'margin-bottom':'24px', 'margin-left': '10px',
                                                      'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)', 'border-radius': '15px', 'display': 'inline-block',
                                                      'background-color':'#f9f9f9', 'padding': '3px'
                                                      })
            ], style={'display':'flex'}
            ),
            dcc.Markdown('''
            ### Tweet Sentiment: Time Series Analysis
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'40%',
                        'marginBottom':5, 'marginTop':15}
            ),
            dcc.Graph(figure=fig_utils.time_fig, style={'width':'90%', 'margin-bottom':'24px', 'margin-left': '10px',
                                              'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)', 'border-radius': '15px', 'display': 'inline-block',
                                              'background-color':'#f9f9f9', 'padding': '3px'
                                              })
        ], 
        ),
        dcc.Tab(label='Preprocessing and Sentiment Classification', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            ### Example Tweet
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.P('Raw Tweet (Edit and see output change):'),
            dcc.Textarea(
                id='textarea-tweet-input',
                value=f"{df_train['OriginalTweet'][10]}",
                style={'width': '50%', 'height': 50},
            ),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Clean Tweet:'),
            dcc.Markdown(id='textarea-tweet-output', 
                     style={'width':'50%','whiteSpace': 'pre-line',
                            'border':'2px darkgrey dashed',
                            'padding':'2px 2px 2px 2px',
                            'marginBottom':5, 'marginTop':15}),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Tokenized (& Padded = 64) Tweet:'),
            dcc.Markdown(id='textarea-tweet-tokenized'),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Prediction for Tweet:'),
            html.Div([
                dcc.Markdown(id='textarea-tweet-prediction'),
                html.Img(
                        src='data:image/png;base64,{}'.format(encoded_horizontal_arrow.decode()),
                        height = '75 px',
                        width = 'auto',
                        style={'padding':'0px 0px 0px 30px'}
                    ),
                dcc.Markdown(id='textarea-tweet-final', style={'padding':'0px 0px 0px 30px'}),
            ], style={'display':'flex'}),
        ], 
        ),
        dcc.Tab(label='Performance Evaluation', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            ### Actual vs Prediction
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.P('Select an Actual Label:'),
            dcc.Dropdown(
                id='actual-vs-prediction-dropdown',
                options =[
                    {'label':i, 'value':i} for i in ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
                         ],
            value='Extremely Negative',
            clearable=False,
            style={'width':'50%'}),
            dcc.Graph(id='actual-vs-prediction-graphic', 
                      style={'width':'83.5%', 'marginBottom':5, 'marginTop':15, 'marginLeft':10,
                             'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)', 'border-radius': '15px', 'display': 'inline-block',
                             'background-color':'#f9f9f9', 'padding': '3px'}),
            dcc.Markdown('''
            ### Backtesting
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'40%',
                        'marginBottom':5, 'marginTop':15}
            ),
            dcc.Graph(figure=actual_pred_time, 
                      style={'width':'90%', 'marginBottom':5, 'marginTop':15, 'marginLeft':15,
                             'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)', 'border-radius': '15px', 'display': 'inline-block',
                             'background-color':'#f9f9f9', 'padding': '3px'})
        ] 
        )
    ], style=tabs_styles
    )
])


@app.callback(
    dash.dependencies.Output('textarea-tweet-output', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = nlp_utils.clean_text(value)
    return value


@app.callback(
    dash.dependencies.Output('textarea-tweet-tokenized', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = nlp_utils.clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    return f'`{value[0]}`'

@app.callback(
    dash.dependencies.Output('textarea-tweet-prediction', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = nlp_utils.clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    value = np.round(model.predict(value), 3)
    labels = ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    return f'`{labels}`  \n + `{value[0]}`'

@app.callback(
    dash.dependencies.Output('textarea-tweet-final', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = nlp_utils.clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    value = np.round(model.predict(value), 3)[0]
    index = value.argmax()
    labels = ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    return f'# **{labels[index]}**'


@app.callback(
    dash.dependencies.Output('actual-vs-prediction-graphic', 'figure'),
    dash.dependencies.Input('actual-vs-prediction-dropdown', 'value'),
)
def update_output(value):
    actual = pd.DataFrame(y_test)         .rename(columns={0:'Extremely Negative', 1:'Negative', 2:'Neutral', 3:'Positive',4:'Extremely Positive'})
    pred = pd.DataFrame(np.round(model.predict(X_test), 0))         .rename(columns={0:1, 1:2, 2:3, 3:4,4:5})
    preds = [1,2,3,4,5]
    actual_vs_pred = pd.concat([actual, pred], axis=1)
    df_plot = actual_vs_pred.loc[actual_vs_pred[value]==1, preds].sum()
    
    
    x=['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    y=df_plot.values
    y_sum = sum(y)
    percentages = ["{:0.2%}".format(v) for v in y / y_sum]
    
    line_colors = ['#D32D41'] * 5
    line_colors[x.index(value)] = '#B3C100'

    fig = go.Figure(go.Bar(
                          x=x, 
                          y=y,
                          marker=dict(
                            color='#1F3F49',
                            line=dict(color=line_colors, width=3))
    ))

    annotations = []
    for x_pos, y_pos, percent in zip(x, y, percentages):
        annotations.append(dict(x=x_pos,
                            y=y_pos + 25,
                            text=f'{percent}',
                            showarrow=False))

    fig.update_layout(yaxis={'title':'Number of Tweets'},
                      xaxis={'title':'Predicted Sentiment of Tweet'},
                      title=f'Model Predictions on Test Data for {value} Tweets',
                      margin={'l':30, 'r':30, 'b':30, 't':30},
                      annotations=annotations)
    return fig


app.run_server(port=8889, host='0.0.0.0')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### backup

# In[ ]:


app = dash.Dash(__name__)


tabs_styles = {
    'height': '50px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '20px',
    'fontWeight': 'bold'
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#6AB187',
    'color': 'black',
    'padding': '20px'
}

vertical_arrow   = 'vertical_arrow.png'
horizontal_arrow = 'horizontal_arrow.png'
encoded_vertical_arrow  = base64.b64encode(open(vertical_arrow, 'rb').read())
encoded_horizontal_arrow  = base64.b64encode(open(horizontal_arrow, 'rb').read())


app.layout = html.Div([
    dcc.Markdown('''
    ## Explaining an LSTM Sentiment Classifier: COVID-19 Twitter Dataset
    ---
    ''', style={'border-left':'7px #1F3F49 solid', 'padding':'2px 0px 0px 20px', 'width':'98%'}
    ),
    dcc.Tabs([
        dcc.Tab(label='Overview', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            > ## Model Overview
            > This dashboard aims to make the inner workings of an LSTM tensorflow model 
            > transparent and easily accessible. The model is trained to classify tweets 
            > related to covid, based on their sentiment. This overview tab mainly focuses
            > on the underlying data that the classification model is built on, see subsequent tabs
            > for information related to preprocessing and modelling.
            
            > The sentiment of a tweet is categorised as one of the following:
            > - Extremely Positive. (5)
            > - Positive. (4)
            > - Neutral. (3)
            > - Negative. (2)
            > - Extremely Negative. (1)
            
            > For access to the dataset see [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv).
            ''', style={'background-color':'#CED2CC', 'color':'black', 
                        'padding':'2px 20px 1px 20px', 'width':'80%', 
                        'marginBottom':5, 'marginTop':10, 'marginLeft':80}
            ),
            dcc.Markdown('''
            ### Sentiment Proportions on Train and Test 
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.Div([
            dcc.Graph(figure=train_proportions, style={'width':'50%'}),
            dcc.Graph(figure=test_proportions, style={'width':'50%'})
            ], style={'display':'flex'}
            ),
            dcc.Markdown('''
            ### Tweet Sentiment: Time Series Analysis
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'40%',
                        'marginBottom':5, 'marginTop':15}
            ),
            dcc.Graph(figure=time_fig, style={'width':'90%'})
            
        ], 
        ),
        dcc.Tab(label='Preprocessing and Sentiment Classification', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            ### Example Tweet
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.P('Raw Tweet (Edit and see output change):'),
            dcc.Textarea(
                id='textarea-tweet-input',
                value=f"{df_train['OriginalTweet'][10]}",
                style={'width': '50%', 'height': 50},
            ),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Clean Tweet:'),
            dcc.Markdown(id='textarea-tweet-output', 
                     style={'width':'50%','whiteSpace': 'pre-line',
                            'border':'2px darkgrey dashed',
                            'padding':'2px 2px 2px 2px',
                            'marginBottom':5, 'marginTop':15}),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Tokenized (& Padded = 64) Tweet:'),
            dcc.Markdown(id='textarea-tweet-tokenized'),
            html.Div(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_vertical_arrow.decode()),
                    height = '100 px',
                    width = '150 px'
                )
            ),
            html.P('Prediction for Tweet:'),
            html.Div([
                dcc.Markdown(id='textarea-tweet-prediction'),
                html.Img(
                        src='data:image/png;base64,{}'.format(encoded_horizontal_arrow.decode()),
                        height = '75 px',
                        width = 'auto',
                        style={'padding':'0px 0px 0px 30px'}
                    ),
                dcc.Markdown(id='textarea-tweet-final', style={'padding':'0px 0px 0px 30px'}),
            ], style={'display':'flex'}),
        ], 
        ),
        dcc.Tab(label='Performance Evaluation', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Markdown('''
            ### Actual vs Prediction
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'50%',
                        'marginBottom':5, 'marginTop':15}
            ),
            html.P('Select an Actual Label:'),
            dcc.Dropdown(
                id='actual-vs-prediction-dropdown',
                options =[
                    {'label':i, 'value':i} for i in ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
                         ],
            value='Extremely Negative',
            clearable=False,
            style={'width':'50%'}),
            dcc.Graph(id='actual-vs-prediction-graphic', 
                      style={'width':'83.5%', 'marginBottom':5, 'marginTop':15, 'marginLeft':10}),
            dcc.Markdown('''
            ### Backtesting
            ''', style={'border-left':'7px #1F3F49 solid', 'padding':'0px 0px 0px 20px', 'width':'40%',
                        'marginBottom':5, 'marginTop':15}
            ),
            dcc.Graph(figure=actual_pred_time, 
                      style={'width':'90%', 'marginBottom':5, 'marginTop':15, 'marginLeft':15})
        ] 
        )
    ], style=tabs_styles
    )
])


@app.callback(
    dash.dependencies.Output('textarea-tweet-output', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = clean_text(value)
    return value


@app.callback(
    dash.dependencies.Output('textarea-tweet-tokenized', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    return f'`{value[0]}`'

@app.callback(
    dash.dependencies.Output('textarea-tweet-prediction', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    value = np.round(model.predict(value), 3)
    labels = ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    return f'`{labels}`  \n + `{value[0]}`'

@app.callback(
    dash.dependencies.Output('textarea-tweet-final', 'children'),
    dash.dependencies.Input('textarea-tweet-input', 'value'),
)
def update_output(value):
    value = clean_text(value)
    value = tokenizer.texts_to_sequences([value])
    value = tf.keras.preprocessing.sequence.pad_sequences(value, max_seqlen)
    value = np.round(model.predict(value), 3)[0]
    index = value.argmax()
    labels = ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    return f'# **{labels[index]}**'


@app.callback(
    dash.dependencies.Output('actual-vs-prediction-graphic', 'figure'),
    dash.dependencies.Input('actual-vs-prediction-dropdown', 'value'),
)
def update_output(value):
    actual = pd.DataFrame(y_test)         .rename(columns={0:'Extremely Negative', 1:'Negative', 2:'Neutral', 3:'Positive',4:'Extremely Positive'})
    pred = pd.DataFrame(np.round(model.predict(X_test), 0))         .rename(columns={0:1, 1:2, 2:3, 3:4,4:5})
    preds = [1,2,3,4,5]
    actual_vs_pred = pd.concat([actual, pred], axis=1)
    df_plot = actual_vs_pred.loc[actual_vs_pred[value]==1, preds].sum()
    
    
    x=['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    y=df_plot.values
    y_sum = sum(y)
    percentages = ["{:0.2%}".format(v) for v in y / y_sum]
    
    line_colors = ['#D32D41'] * 5
    line_colors[x.index(value)] = '#B3C100'

    fig = go.Figure(go.Bar(
                          x=x, 
                          y=y,
                          marker=dict(
                            color='#1F3F49',
                            line=dict(color=line_colors, width=3))
    ))

    annotations = []
    for x_pos, y_pos, percent in zip(x, y, percentages):
        annotations.append(dict(x=x_pos,
                            y=y_pos + 25,
                            text=f'{percent}',
                            showarrow=False))

    fig.update_layout(yaxis={'title':'Number of Tweets'},
                      xaxis={'title':'Predicted Sentiment of Tweet'},
                      title=f'Model Predictions on Test Data for {value} Tweets',
                      margin={'l':30, 'r':30, 'b':30, 't':30},
                      annotations=annotations)
    return fig


app.run_server(port=8889, host='0.0.0.0')

