import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np


class sentimentTweets(Dataset):
    def __init__(self):
        self.train_df = pd.read_csv("src/data/train.csv")
        self.test_df = pd.read_csv("src/data/test.csv")

        self.train_max_len = max([len(str(sent)) for sent in self.train_df.text])
        self.test_max_len = max([len(str(sent)) for sent in self.test_df.text])
        #print("train max_length: ", self.train_max_len)
        #print("test max_length: ", self.test_max_len)

        #pretext
        possible_labels = ['negative','neutral', 'positive']
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index

        self.train_df['label'] = self.train_df.sentiment.replace(label_dict)
        self.test_df['label'] = self.test_df.sentiment.replace(label_dict)

        for i,sent in enumerate(self.train_df["text"]):
            self.train_df.at[i,"text"] = self.prepare_data(str(sent))
        for i,sent in enumerate(self.test_df["text"]):
            self.test_df.at[i,"text"] = self.prepare_data(str(sent))

    def prepare_data(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

if __name__ == "__main__":
    data = sentimentTweets()
    