import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split


class CNNModel(tf.keras.Model):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', strides=(1, 1), input_shape=(25, 25, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', strides=(1, 1), input_shape=(25, 25, 1)),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, inputs):
        delta = inputs[0]
        stop_state = inputs[1]
        for _ in range(1): # How to include delta?
            stop_state = self.model(stop_state)
        return stop_state

def convert_1d_to_3d(x):
    return x.reshape((x.shape[0], 25, 25, 1))

def convert_3d_to_1d(x):
    return x.reshape((x.shape[0], 625))

def get_train_dataset():
    train = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')
    deltas = train.iloc[:, 1].values.reshape((train.shape[0], 1))
    end_states = train.iloc[:, 627:].values
    start_states = train.iloc[:, 2:627].values
    train_x, eval_x, train_y, eval_y = train_test_split(
        np.concatenate([deltas, end_states], axis=1),
        start_states,
        test_size=0.33,
        random_state=42)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_x[:, 0], convert_1d_to_3d(train_x[:, 1:])),  # inputs
         convert_1d_to_3d(train_y)))  # targets
    eval_dataset = tf.data.Dataset.from_tensor_slices((
        (eval_x[:, 0], convert_1d_to_3d(eval_x[:, 1:])),  # inputs
         convert_1d_to_3d(eval_y)))  # targets
    return train_dataset.shuffle(3).batch(128), eval_dataset.batch(128)

def get_test_dataset():
    test = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv')
    deltas = test.iloc[:, 1].values
    end_states = test.iloc[:, 2:].values
    test_dataset = tf.data.Dataset.from_tensor_slices((
        (deltas, convert_1d_to_3d(end_states)),  # inputs
        convert_1d_to_3d(end_states)))  # fake targets
    return test_dataset.batch(128)

if __name__ == '__main__':
    train_dataset, eval_dataset = get_train_dataset()
    model = CNNModel()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    model.fit(train_dataset, epochs=10, validation_data=eval_dataset)
    test_dataset = get_test_dataset()
    pred = model.predict(test_dataset)
    board = (pred > 0.5).astype(int)
    submission = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/sample_submission.csv', index_col='id')
    submission[:] = convert_3d_to_1d(board)
    submission.to_csv('submission.csv')