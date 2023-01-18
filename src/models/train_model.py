import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import pandas as pd
import numpy as np

from src.models.model import TwitterSentimentAnalysis

def main():

    train_df = pd.read_csv("src/data/train.csv")
    test_df = pd.read_csv("src/data/test.csv")

    #pretext
    print(train_df[0])

    ## GPU
    gpus = 0
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")

    ## Dataloader

    ## Train


if __name__ == "__main__":
    main()
    