import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from scipy.special import softmax

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import pandas as pd
import numpy as np

from src.models.model import TwitterSentimentAnalysis
from src.data.make_dataset import sentimentTweets


def main():
    ## load data
    print("Loading data...")
    data = sentimentTweets()
    test_df = data.test_df.head(100)

    ##load model
    print("Loading model...")
    model = TwitterSentimentAnalysis()

    ##encoder
    print("Encoding tokens...")
    encoded_input = model.tokenizer.batch_encode_plus(test_df["text"], 
                                                    return_tensors='pt',
                                                    return_attention_mask=True,
                                                    padding='max_length',
                                                    max_length=data.train_max_len)

    ## Test
    total_correct = 0
    total_wrong = 0

    output = model(**encoded_input)
    scores = output.detach().numpy()
    scores = softmax(scores)
    for i,score in enumerate(scores):
        if np.argmax(score)==test_df["label"][i]:
            total_correct += 1
        else:
            total_wrong += 1
    
    print(total_wrong,total_correct)

if __name__ == "__main__":
    main()
    