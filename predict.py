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

# globals
model = None

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()


def predict_batch(batch_id, data_path="CC_data/", save_path="output/"):
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=3, cache_dir=None)
    # sentence_pred_df = pd.DataFrame()
    sentence_pred_df = []
    news_id = 0

    start_main = time.time()

    data = pickle.load(open(data_path + "BERTnews{}.p".format(batch_id), "rb"))
    data = data.reset_index(drop=True)

    for i in range(len(data)):
        pred = predict(data.loc[i]['text'], data.loc[i]['index'], model, write_to_csv=False)
        sentence_pred_df.extend(pred)
        news_id += 1

    sentence_pred_df = pd.DataFrame.from_dict(sentence_pred_df)
    sentence_pred_df.to_csv(save_path + "BERTnews_preds_{}.csv".format(batch_id))

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


if __name__=="__main__":

    # ========= single prediction =========
    # start = time.time()
    # predict_batch(0)
    # end = time.time()
    # print("TOTAL time: {}".format(round(end-start, 2)))

    # ======== New multiprocessing ===========

    # N = 539317
    N = 5000
    # N = 30

    # we parse data to list of tuples to avoid reloading entire data for every subprocess
    data = pickle.load(open("CC_data/BERTnews_all.p", "rb"))
    data = [tuple(x) for x in data.head(N).itertuples(index=False)]

    pool = mp.Pool(initializer=init_bert)
    print("Number of cores: ", os.cpu_count())

    start = time.time()
    res = pool.map(predict_news, data)
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










