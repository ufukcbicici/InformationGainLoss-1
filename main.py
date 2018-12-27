import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from sequences import FashionMnistSequence
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import json
from mongo_handler import MongoHandler
from loss_function import InformationGainLoss


mongo_handler = MongoHandler()
experiment_id = mongo_handler.get_experiment_id()

### Training Settings
training_parameters = {
    "dataset": "fashion mnist",
    "model_log_dir": f"./logs/{experiment_id}",
    "model_save_file": f"./models/model_{experiment_id}.model",
    "validation_split": 0.1,
    "batch_size": 100,
    "epochs": 2,
    "dropout": 0.65,
    "optimizer": {"name": "adam",
                  "learning_rate": 0.0001,
                  "decay": 0.,
                  }
}

pathlib.Path(training_parameters["model_log_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.dirname(training_parameters["model_save_file"])).mkdir(parents=True, exist_ok=True)

### Prepare Data

VALIDATION_SPLIT = training_parameters["validation_split"]
BATCH_SIZE = training_parameters["batch_size"]

(x_data, y_data), (x_test, y_test) = fashion_mnist.load_data()
shuffled_indexes = np.random.permutation(np.arange(len(x_data)))

TRAIN_INSTANCES = int((1 - VALIDATION_SPLIT) * len(x_data))
TEST_INSTANCES = len(x_test)
VALIDATION_INSTANCES = int(VALIDATION_SPLIT * len(x_data))

x_train = x_data[shuffled_indexes[:TRAIN_INSTANCES]]
y_train = y_data[shuffled_indexes[:TRAIN_INSTANCES]]

x_validation = x_data[shuffled_indexes[TRAIN_INSTANCES:]]
y_validation = y_data[shuffled_indexes[TRAIN_INSTANCES:]]

num_of_classes = len(np.unique(y_data))

train_data_generator = FashionMnistSequence(x_train, y_train, BATCH_SIZE, num_of_classes, name='Train')
validation_data_generator = FashionMnistSequence(x_validation, y_validation, BATCH_SIZE, num_of_classes,
                                                 name='validation')
test_data_generator = FashionMnistSequence(x_test, y_test, BATCH_SIZE, num_of_classes, name='test')

### Prepare Architecture

inputs = keras.layers.Input(shape=(28, 28, 1))

conv_1 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same')(inputs)
relu_1 = keras.layers.ReLU()(conv_1)
maxpool_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(relu_1)

conv_2 = keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same')(maxpool_1)
relu_2 = keras.layers.ReLU()(conv_2)
maxpool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(relu_2)

conv_3 = keras.layers.Conv2D(128, kernel_size=(1, 1), padding='same')(maxpool_2)
relu_3 = keras.layers.ReLU()(conv_3)
maxpool_3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(relu_3)

flatten = keras.layers.Flatten()(maxpool_3)

dense_4 = keras.layers.Dense(1028, )(flatten)
relu_4 = keras.layers.ReLU()(dense_4)
dropout_4 = keras.layers.Dropout(rate=training_parameters["dropout"])(relu_4)

dense_5 = keras.layers.Dense(512)(dropout_4)
relu_5 = keras.layers.ReLU()(dense_5)
dropout_5 = keras.layers.Dropout(rate=training_parameters["dropout"])(relu_5)

dense_6 = keras.layers.Dense(10)(dropout_5)
softmax_6 = keras.layers.Softmax()(dense_6)

# Organize Inputs and Outputs
model = keras.Model(inputs=[inputs],
                    outputs=[softmax_6])

# Optimizer
if training_parameters["optimizer"]["name"].lower() == "adam":
    optimizer = keras.optimizers.Adam(lr=training_parameters["optimizer"]["learning_rate"],
                                      decay=training_parameters["optimizer"].get("decay", 0.))

if training_parameters["optimizer"]["name"].lower() == "sgd":
    optimizer = keras.optimizers.SGD(lr=training_parameters["optimizer"]["learning_rate"],
                                     momentum=training_parameters["optimizer"].get("momentum", 0.),
                                     decay=training_parameters["optimizer"].get("decay", 0.))

model.compile(optimizer=optimizer,
              loss=[InformationGainLoss.get_loss_function(maxpool_1, maxpool_2, maxpool_3)],
              metrics=[keras.metrics.categorical_accuracy]
              )

hist = model.fit_generator(generator=train_data_generator,
                           epochs=training_parameters["epochs"],
                           validation_data=validation_data_generator,
                           validation_steps=len(validation_data_generator),
                           callbacks=[keras.callbacks.TensorBoard(log_dir=training_parameters["model_log_dir"],
                                                                  write_images=True),
                                      keras.callbacks.ModelCheckpoint(filepath=training_parameters["model_save_file"]),
                                      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

results = model.evaluate_generator(test_data_generator)

model_dict = model.get_config()
training_parameters["test_loss"] = results[0]
training_parameters["test_metrics"] = results[1]
training_parameters["history"] = hist.history
training_parameters["model"] = json.loads(model.to_json())


mongo_handler.save_experiment(training_parameters)
