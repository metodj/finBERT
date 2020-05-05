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


parser = argparse.ArgumentParser(description='Sentiment analyzer')

parser.add_argument('-a', action="store_true", default=False)

parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, help='Where to write the results')
parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


# with open(args.text_path,'r', encoding='utf-8') as f:
#     text = f.read()
#
# model = BertForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)
# #now = datetime.datetime.now().strftime("predictions_%B-%d-%Y-%I:%M.csv")
# random_filename = ''.join(random.choice(string.ascii_letters) for i in range(10))
# output = random_filename + '.csv'
# predict(text, model,write_to_csv=True,path=os.path.join(args.output_dir,output))

def predict_batch(batch_id, data_path="CC_data/", save_path="output/"):
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=3, cache_dir=None)
    # sentence_pred_df = pd.DataFrame()
    sentence_pred_df = []
    news_id = 0

    start_main = time.time()

    data = pickle.load(open(data_path + "BERTnews{}.p".format(batch_id), "rb"))
    data = data.reset_index(drop=True)
    N = 30

    for i in range(N):
        pred = predict(data.loc[i]['text'], data.loc[i]['index'], model, write_to_csv=False)
        sentence_pred_df.extend(pred)
        news_id += 1

    sentence_pred_df = pd.DataFrame.from_dict(sentence_pred_df)
    sentence_pred_df.to_csv(save_path + "BERTnews_preds_{}.csv".format(batch_id))

    end_main = time.time()
    print("TIME for batch_id: {}".format(round(end_main - start_main, 2)))

if __name__=="__main__":

    pool = mp.Pool()
    print("Number of cores: ", os.cpu_count())

    start = time.time()
    pool.map(predict_batch, list(range(2)))
    end = time.time()
    print("TOTAL time: {}".format(round(end-start, 2)))


