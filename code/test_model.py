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
bert_layer = hub.KerasLayer(bert_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

test = pd.read_csv("../data/input/test.csv")
submission = pd.read_csv("../data/input/sample_submission.csv")

test_input = bert_input(test.text.values, tokenizer, max_len=256)

model = bert_model(bert_layer, max_len=256)

model.load_weights('../data/output/model/model.h5')
test_pred = model.predict(test_input)

submission['target'] = test_pred.round().astype(int)
submission.to_csv('../data/output/submission.csv', index=False)