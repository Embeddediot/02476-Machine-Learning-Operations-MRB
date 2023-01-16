## Group Members
Michelle Langkjær S153520\
Bjørn Sejer S172533\
Rasmus Ulstrup S173920
# 02476-Machine-Learning-Operations-MRB
## Overall goal of the project
The project is inspired by a kaggle competition called Tweet Sentiment Extraction. The goal is to develop a model which can classify tweets into positive, neutral or negative.
## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
As we are dealing a natural language processing (NLP) problem, it is only natrual to use the Transformers framwork.
## How to you intend to include the framework into your project
To include PyTorch Transformers in our NLP project, we would first select a pre-trained language model from the library, fine-tune it on our dataset, preprocess the tweets, and use the model to make predictions. We would then evaluate the model's performance and make any necessary adjustments.
## What data are you going to run on (initially, may change)
Our dataset is from kaggle: [Tweet sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview/description). The data is split into train, test and sample_submission. Each row contains the text of a tweet and a sentiment label. In the training set you are provided with a word or phrase drawn from the tweet (selected_text) that encapsulates the provided sentiment.
## What deep learning models do you expect to use
For this project pre-trained models will be used as the deadline will not allow time to train models. 
For this project the model BERTweet will be used. It is a pre-trained language model for English Tweets which allows us to train it on our dataset without training it from zero. It is perfect for this project since the data will consist of English tweets.
If time allows it we will also compare the results with other models.
