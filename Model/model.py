from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras


class Model:
    def __init__(self):
        self.path = Path('./model')
        self.load_preprocessor()
        # self.load_model()

    def load_preprocessor(self):
        self.preprocessor = hub.KerasLayer(
            "https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_preprocess/3")

    def load_model(self):
        self.predictor = keras.models.load_model(self.path / 'bert.h5',
            custom_objects={'KerasLayer': hub.KerasLayer})

    def preprocess(self, text):
        return self.preprocessor(text)

    def predict(self, text):
        t = self.preprocess([text])
        pred = self.predictor.predict(t)[0]
        print(pred)
        tensor = tf.math.argmax(pred)
        label = tensor.numpy()

        return label
