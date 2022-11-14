import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import numpy as np
import pandas as pd

import tokenization
from bert import bert_input
from bert import bert_model

bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
#bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
#bert_url = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"
#bert_url = "https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/1"
#bert_url = "https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/1"
#bert_url = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(bert_url, trainable=True)

train = pd.read_csv("../data/input/train.csv")

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# tokenizer = tokenization.FullTokenizer(vocab_file)

train_input = bert_input(train.text.values, tokenizer, max_len=256)
train_labels = train.target.values

model = bert_model(bert_layer, max_len=256)
checkpoint = ModelCheckpoint('../data/output/model/model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)

