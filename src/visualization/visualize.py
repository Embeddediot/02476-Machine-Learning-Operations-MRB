import pandas as pd
import numpy as np

path = r'C:\Users\uller\OneDrive\Dokumenter\DeepLearning\02476-Machine-Learning-Operations-MRB\data'
train = 'train.csv'
test = 'test.csv'

train_df = pd.read_csv(f'C:\\Users\\uller\\OneDrive\\Dokumenter\\DeepLearning\\02476-Machine-Learning-Operations-MRB\\data\\train.csv')
test_df = pd.read_csv(f'C:\\Users\\uller\\OneDrive\\Dokumenter\\DeepLearning\\02476-Machine-Learning-Operations-MRB\\data\\test.csv')

print(train_df.sample(10))