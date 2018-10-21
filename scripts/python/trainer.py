from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import numpy as np

def text_to_data(text, author):
    #remove newlines, numbers, some punctuation
    text = text.replace('\n', " ")
    text = re.sub(r'[0-9]+', '', text)
    sent_tokenize_list = sent_tokenize(text)    
    total_arr = [(x, author) for x in sent_tokenize_list]
    #total_dict = {"text" : [x for x in sent_tokenize_list], "author" : [author for x in sent_tokenize_list]}
    #total_dict = pd.DataFrame(data=total_dict)
    vocab_count = len(set(text.split(' ')))
    return total_arr, vocab_count

def shuffle_and_convert(dictionary):
    [[x, y] for x in dictionary['text'] for y in dictionary['author']]

total_word_count = 0
f = open('../../data/anthem.txt', 'r')
anthem_text, count = text_to_data(f.read(), "ayn rand")
total_word_count += count

f = open('../../data/mobydick.txt', 'r')
mobydick_text, count = text_to_data(f.read(), "herman melville")
total_word_count += count

f = open('../../data/alice.txt', 'r')
alice_text, count = text_to_data(f.read(), 'lewis carroll')
total_word_count += count

total_arr = anthem_text + mobydick_text + alice_text
np.random.seed(53894343)
np.random.shuffle(total_arr)

total_dict = {"text" : [x[0] for x in total_arr], "author" : [x[1] for x in total_arr]}
total_df = pd.DataFrame(total_dict)
print(total_df.describe())

from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder
from keras import utils
from keras.layers import Dense, Flatten, LSTM, Embedding, Dropout

def tokenize_input(train, test, vocab):
    # tokenize input text
    tokenizer = text.Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(train)
    # create input matrix
    return tokenizer.texts_to_matrix(train), \
           tokenizer.texts_to_matrix(test), \
           tokenizer

def encode_labels(train, test):
    encoder = LabelEncoder()
    encoder.fit(train)
    y_train = encoder.transform(train)
    num_classes = np.max(y_train) + 1
    return utils.to_categorical(y_train, num_classes), \
           utils.to_categorical(encoder.transform(test), num_classes), \
           encoder.classes_, num_classes

def split_dataset(dataset, split, input_field, label_field, ):
    train_size = int(len(dataset[input_field]) * split)
    return dataset[input_field][:train_size], dataset[input_field][train_size:], \
           dataset[label_field][:train_size], dataset[label_field][train_size:]

def init_data(data, vocab_size, data_split):
    train_input, test_input, train_label, test_label = split_dataset(total_dict, data_split, 'text', 'author')
    x_train, x_test, tokenizer = tokenize_input(train_input, test_input, vocab_size)
    y_train, y_test, text_labels, num_classes = encode_labels(train_label, test_label)
    return train_input, test_input, train_label, test_label, x_train, x_test, y_train, y_test, tokenizer, text_labels,\
           num_classes

def conv_x(xval, tokenizer):
    ret = tokenizer.texts_to_matrix(xval)
    return ret

def avgPrediction(predictionsArr):
    predictionAvgs = []
    average = 0;
    for x in range(len(predictionsArr[0])):
        for y in range(len(predictionsArr)):
            average += predictionsArr[y][x] / len(predictionsArr)
        predictionAvgs.append(average)
    # print it
            

train_input, test_input, train_label, test_label, x_train, x_test, y_train, y_test, tokenizer, text_labels, \
    num_classes = init_data(total_df, total_word_count, .7)

#pred_x = conv_x("Well, if I must, I must,’ the King said, with a melancholy air", tokenizer)
#pred_x = conv_x("Well, well, well! Stubb knows him best of all, and Stubb always says he’s queer; says nothing but that one sufficient little word queer; he’s queer, says Stubb; he’s queer—queer, queer; and keeps dinning it into Mr. Starbuck all the time—queer—sir—queer, queer, very queer. And here’s his leg! Yes, now that I think of it, here’s his bedfellow! has a stick of whale’s jaw-bone for a wife! And this is his leg; he’ll stand on this. What was that now about one leg standing in three places, and all three places standing in one hell—how was that? Oh! I don’t wonder he looked so scornful at me! I’m a sort of strange-thoughted sometimes, they say; but that’s only haphazard-like. Then, a short, little old body like me,", tokenizer)
pred_x = conv_x("Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore", tokenizer)

print(x_train.shape)
model = Sequential([
    Embedding(total_word_count, 32),
    LSTM(3),
    Dropout(.2),
    Dense(512, input_shape=(total_word_count,)),
    Dense(3, activation='sigmoid')
    #Dense(512),
    #Activation('sigmoid'), #TODO: change to sigmoid?
    #Dense(num_classes),
    #Activation('softmax')
])
model.compile(loss='categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.05)

pred = model.predict(pred_x)

#scores = model.evaluate(x_test, y_test, verbose=1)
#print(scores)

print(pred)
#print("Accuracy: %.2f%%" % (scores[1] * 100))
#model.fit(x_train,y_train,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);
#print(model.summary())