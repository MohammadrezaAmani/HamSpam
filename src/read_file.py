import pandas as pd
import os

def _create_dataset(path, label):
  files = [file for file in os.listdir(path)]
  data = list()
  for file in files:
    with open(path +"/" + file, "r" , encoding='utf-8') as f:
      temp = f.read()
      data.append(temp)

  df = pd.DataFrame()
  df["Text"] = data
  df["label"] = label
  return df

def create_dataset():

  train = pd.concat([_create_dataset('/home/alireza/Desktop/src/emails/hamtraining/', "ham"),  _create_dataset('/home/alireza/Desktop/src/emails/spamtraining/', "spam")] , axis=0)
  test = pd.concat([_create_dataset('/home/alireza/Desktop/src/emails/hamtesting/', "ham"), _create_dataset('/home/alireza/Desktop/src/emails/spamtesting/', "spam")], axis=0)

  train["target"] = train["label"].replace(["spam", "ham"], [1, 0])
  test["target"] = test["label"].replace(["spam", "ham"], [1, 0])
  df = pd.concat([test, train ], axis=0)
  return df
