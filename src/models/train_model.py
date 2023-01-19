import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

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

    ##load model
    print("Loading model...")
    model = TwitterSentimentAnalysis()

    ## GPU
    gpus = 0
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")

    ##encoder
    print("Encoding tokens...")
    encoded_input = model.tokenizer.batch_encode_plus(data.train_df["text"], 
                                                    return_tensors='pt',
                                                    return_attention_mask=True,
                                                    padding='max_length',
                                                    max_length=data.train_max_len)
    
    #train set
    input_ids_train = encoded_input['input_ids']
    attention_masks_train = encoded_input['attention_mask']
    labels_train = torch.tensor(data.train_df.label.values)

    ## Dataloader
    #train set
    dataset_train = TensorDataset(input_ids_train, 
                                  attention_masks_train,
                                  labels_train)

    batch_size = 32

    dataloader_train = DataLoader(dataset_train,
                              sampler = RandomSampler(dataset_train),
                              batch_size = batch_size,
                              num_workers = os.cpu_count() or 2)

    ## Train
    print("Training...")
    trainer = Trainer(
        max_epochs=5,
        gpus=gpus
    )

    trainer.fit(
        model,
        dataloader_train
    )


if __name__ == "__main__":
    main()
    