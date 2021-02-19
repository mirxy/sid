import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop = stopwords.words('english')
wnl = WordNetLemmatizer()


class LSTM:
    def __init__(self):
        self.max_len = 50
        self.path = Path('./LSTM/model')
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self):
        print(self.path.absolute())
        with open(self.path / 'tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)

    def load_model(self):
        self.predictor = keras.models.load_model(self.path / 'LSTM.h5')

    def preprocess(self, text):
        t = text.lower()
        t = word_tokenize(t)

        tmp = []
        for w in t:
            if w not in stop:
                w = wnl.lemmatize(w)
                if len(w) >= 3:
                    tmp.append(w)

        return ' '.join(tmp)

    def predict(self, text):
        t = self.preprocess(text)
        t = self.tokenizer.texts_to_sequences([t])
        t = pad_sequences(t, maxlen=self.max_len)

        pred = self.predictor.predict(t)[0]
        print(pred)
        tensor = tf.math.argmax(pred)
        label = tensor.numpy()

        return label
