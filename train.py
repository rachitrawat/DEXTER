import logging

import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils

from keras_lib.WideResNet import WideResNet
from utils import load_data

logging.basicConfig(level=logging.DEBUG)


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


# path to db.mat
db_path = "data/imdb_db.mat"
# batch size
batch_size = 16
# number of epochs
epochs = 20
# depth of network (10, 16, 22, 28...)
depth = 16
# width of network
k = 8
# validation split ratio
validation_split = 0.1
# checkpoints path
output_path = "checkpoints"

logging.debug("Loading data...")
image, gender, age, _, image_size, _ = load_data(db_path)
X_data = image
y_data_g = np_utils.to_categorical(gender, 2)
y_data_a = np_utils.to_categorical(age, 101)

model = WideResNet(image_size, depth=depth, k=k)()
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=["categorical_crossentropy", "categorical_crossentropy"],
              metrics=['accuracy'])

logging.debug("Model summary...")
model.count_params()
model.summary()

callbacks = [LearningRateScheduler(schedule=Schedule(epochs)),
             ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                             monitor="val_loss",
                             verbose=1,
                             save_best_only=True,
                             mode="auto")
             ]

logging.debug("Training...")

data_num = len(X_data)
indexes = np.arange(data_num)
np.random.shuffle(indexes)
X_data = X_data[indexes]
y_data_g = y_data_g[indexes]
y_data_a = y_data_a[indexes]
train_num = int(data_num * (1 - validation_split))
X_train = X_data[:train_num]
X_test = X_data[train_num:]
y_train_g = y_data_g[:train_num]
y_test_g = y_data_g[train_num:]
y_train_a = y_data_a[:train_num]
y_test_a = y_data_a[train_num:]

hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                 validation_data=(X_test, [y_test_g, y_test_a]))

logging.debug("Saving history...")
pd.DataFrame(hist.history).to_hdf("history_{}_{}.h5".format(depth, k), "history")
