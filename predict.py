from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import argparse
from pathlib import Path
import datetime
import os
import random
import string
import pandas as pd
import time
import pickle
import multiprocessing as mp
import gc

# globals
model = None

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()


def predict_batch(N, data_path="CC_data/", save_path="output/"):
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=3, cache_dir=None)
    sentence_pred_df = []

    start_main = time.time()

    data = pickle.load(open(data_path + "BERTnews_all.p", "rb"))
    data = data.reset_index(drop=True)

    # for i in range(len(data)):
    for i in range(N):
        pred = predict(data.loc[i]['text'], data.loc[i]['index'], model, write_to_csv=False)
        sentence_pred_df.extend(pred)

    sentence_pred_df = pd.DataFrame.from_dict(sentence_pred_df)
    sentence_pred_df.to_csv(save_path + "BERTnews_preds.csv")

    end_main = time.time()
    print("TIME for batch_id: {}".format(round(end_main - start_main, 2)))


def init_bert(model_path=args.model_path):
    global model
    # global data
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3, cache_dir=None)
    # data = pickle.load(open("CC_data/BERTnews_all.p", "rb"))


def predict_news(x):
    pred = predict(x[1], x[0], model, write_to_csv=False)
    return pred


if __name__ == "__main__":

    # ========= single prediction =========
    # start = time.time()
    # predict_batch(30)
    # end = time.time()
    # print("TOTAL time: {}".format(round(end-start, 2)))

    # ======== New multiprocessing ===========

    N_start = 0
    # N_end = 539317
    # N_end = 5000
    # N_end = 30
    N_end = 100000

    # we parse data to list of tuples to avoid reloading entire data for every subprocess
    data = pickle.load(open("CC_data/BERTnews_all.p", "rb"))
    data_batch = [tuple(x) for x in data.loc[N_start:N_end].itertuples(index=False)]

    del data
    gc.collect()

    pool = mp.Pool(initializer=init_bert)
    print("Number of cores: ", os.cpu_count())

    start = time.time()
    res = pool.map(predict_news, data_batch)
    end = time.time()
    print("TOTAL time: {}".format(round(end-start, 2)))

    # save to pandas dataframe
    flatten = lambda l: [item for sublist in l for item in sublist]
    res = flatten(res)
    res = pd.DataFrame.from_dict(res)
    res.to_csv("output/BERTnews_preds_all.csv")

    # ========= Naive multiprocessing =========

    # pool = mp.Pool()
    # print("Number of cores: ", os.cpu_count())
    #
    # start = time.time()
    # pool.map(predict_batch, list(range(2)))
    # end = time.time()
    # print("TOTAL time: {}".format(round(end-start, 2)))










