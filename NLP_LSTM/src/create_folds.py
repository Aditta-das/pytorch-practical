import pandas as pd 
import numpy as np 
from sklearn import model_selection
from tqdm import tqdm
if __name__ == "__main__":
    df = pd.read_csv("G:/approaching by abhishek/NLP_LSTM/input/imdb.csv")
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    df.to_csv("G:/approaching by abhishek/NLP_LSTM/input/imdb_folds.csv", index=False)
    print("[INFO] Fold finished")