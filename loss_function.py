import tensorflow as tf
from tensorflow import keras


class InformationGainLoss:
    @staticmethod
    def get_loss_function(*layers):
        def loss(y_true, y_pred):
            """

            :param y_true: ground-truth
            :param y_pred: output of convolution
            :return:
            """
            for layer in layers:
                print(layer)
            print(y_true)
            print(y_pred)

            return keras.losses.categorical_crossentropy(y_true, y_pred)

        return loss
