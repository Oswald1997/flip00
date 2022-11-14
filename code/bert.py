import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import pickle

def bert_input(texts, tokenizer, max_len=512):
    tokens_list = []
    masks_list = []
    segments_list = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        tokens_list.append(tokens)
        masks_list.append(pad_masks)
        segments_list.append(segment_ids)
        
        tokens_matrix = np.array(tokens_list)
        tokens_masks = np.array(masks_list)
        tokens_segments = np.array(segments_list)

    return tokens_matrix, tokens_masks, tokens_segments




def bert_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def check_bert_input():
    f1 = open('../data/output/token.data', 'rb')
    tokenlist = pickle.load(f1)
    print(tokenlist[10])
    f1.close()

    f2 = open('../data/output/mask.data', 'rb')
    masklist = pickle.load(f2)
    print(masklist[10])
    f2.close()

    f3 = open('../data/output/segment.data', 'rb')
    segmentlist = pickle.load(f3)
    print(segmentlist[10])
    f3.close()

if __name__ == '__main__':
     check_bert_input()