from tensorflow import keras
import numpy as np


class FashionMnistSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, num_of_classes, name=""):
        self.x, self.y = x_set, y_set
        self.num_of_classes = num_of_classes
        self.batch_size = batch_size
        self.name = name

    def __len__(self):
        return int(len(self.x) / float(self.batch_size))

    def __getitem__(self, item):
        print(f"[INFO] {self.name} generator has been called for batch number {item + 1}")
        x_batch = self.x[item * self.batch_size:(item + 1) * self.batch_size]
        y_batch = self.y[item * self.batch_size:(item + 1) * self.batch_size]
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], x_batch.shape[2], 1)
        x_batch = x_batch / 255.0
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_of_classes)

        return x_batch, y_batch
