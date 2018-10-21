# coding: utf-8
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder
from keras import utils
from keras.layers import Dense, Flatten, LSTM, Embedding, Dropout
import json

class MyServer(Flask):

    def __init__(self, *args, **kwargs):
        super(MyServer, self).__init__(*args, **kwargs)
        total_word_count = 0
        f = open('../../data/anthem.txt', 'r')
        anthem_text, count = self.text_to_data(f.read(), "ayn rand")
        total_word_count += count

        f = open('../../data/mobydick.txt', 'r')
        mobydick_text, count = self.text_to_data(f.read(), "herman melville")
        total_word_count += count

        f = open('../../data/alice.txt', 'r')
        alice_text, count = self.text_to_data(f.read(), 'lewis carroll')
        total_word_count += count

        total_arr = anthem_text + mobydick_text + alice_text
        np.random.seed(53894343)
        np.random.shuffle(total_arr)

        self.total_dict = {"text" : [x[0] for x in total_arr], "author" : [x[1] for x in total_arr]}
        total_df = pd.DataFrame(self.total_dict)
        #print(total_df.describe())
        self.train_input, self.test_input, self.train_label, self.test_label, self.x_train, self.x_test, self.y_train, self.y_test, self.tokenizer, self.text_labels, self.num_classes = self.init_data(total_df, total_word_count, .7)
        model = Sequential([
            Dense(512),
            Activation('sigmoid'),
            Dense(self.num_classes),
            Activation('softmax')
        ])
        model.compile(loss='categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.05)
        model._make_predict_function()
        self.model = model

    def text_to_data(self, text, author):
        #remove newlines, numbers, some punctuation
        text = text.replace('\n', " ")
        text = re.sub(r'[0-9]+', '', text)
        sent_tokenize_list = sent_tokenize(text)    
        total_arr = [(x, author) for x in sent_tokenize_list]
        vocab_count = len(set(text.split(' ')))
        return total_arr, vocab_count

    def shuffle_and_convert(self, dictionary):
        [[x, y] for x in dictionary['text'] for y in dictionary['author']]

    def tokenize_input(self, train, test, vocab):
        # tokenize input text
        self.tokenizer = text.Tokenizer(num_words=vocab)
        self.tokenizer.fit_on_texts(train)
        # create input matrix
        return self.tokenizer.texts_to_matrix(train), self.tokenizer.texts_to_matrix(test), self.tokenizer

    def encode_labels(self, train, test):
        encoder = LabelEncoder()
        encoder.fit(train)
        y_train = encoder.transform(train)
        num_classes = np.max(y_train) + 1
        return utils.to_categorical(y_train, num_classes), utils.to_categorical(encoder.transform(test), num_classes), encoder.classes_, num_classes

    def split_dataset(self, dataset, split, input_field, label_field, ):
        train_size = int(len(dataset[input_field]) * split)
        return dataset[input_field][:train_size], dataset[input_field][train_size:], dataset[label_field][:train_size], dataset[label_field][train_size:]

    def init_data(self, data, vocab_size, data_split):
        train_input, test_input, train_label, test_label = self.split_dataset(self.total_dict, data_split, 'text', 'author')
        x_train, x_test, tokenizer = self.tokenize_input(train_input, test_input, vocab_size)
        y_train, y_test, text_labels, num_classes = self.encode_labels(train_label, test_label)
        return train_input, test_input, train_label, test_label, x_train, x_test, y_train, y_test, tokenizer, text_labels, num_classes

    def conv_x(self, xval, tokenizer):
        ret = tokenizer.texts_to_matrix(xval)
        return ret

    def avgPrediction(self, predictionsArr):
        predictionAvgs = []
        average = 0;
        for x in range(len(predictionsArr[0])):
            for y in range(len(predictionsArr)):
                average += predictionsArr[y][x] / len(predictionsArr)
            predictionAvgs.append(average)
            average = 0
        return predictionAvgs

app = MyServer(__name__)
@app.route('/confidence', methods=['POST'])
def returnConfidence():
    data = request.json
    print(request.json)
    pred_x = app.conv_x(data['body']['exerpt'], app.tokenizer)
    pred = app.model.predict(pred_x)
    return_data = [
        {"author": "Ayn Rand", "confidence": str(pred[0][0])},
        {"author": "Herman Melville", "confidence": str(pred[0][1])},
        {"author": "Lewis Carroll", "confidence": str(pred[0][2])}
    ]
    return jsonify("{" + str(return_data) +"}")

