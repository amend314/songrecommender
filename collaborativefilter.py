import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split


csvList = []

for files in os.listdir('data'):
    df = pd.read_csv('data/' + files, sep=',', header=0)
    csvList.append(df)

df = pd.concat(csvList, axis=0, ignore_index=True)
df = df.drop(columns=['artist_name', 'track_name'])
df_train, df_test = train_test_split(df, test_size=0.2)

