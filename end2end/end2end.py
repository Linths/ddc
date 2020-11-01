import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
from tensorflow.keras import Model
import numpy as np
import pickle
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from os import listdir
from os.path import isfile, join

from util import *
from util2 import *

np.set_printoptions(precision=3)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

DO_TRAIN = True
WEIGHTS_FILE = '/content/gdrive/My Drive/ddr_weights_long'
TRAIN_DIR = 'data/data_split/train'
TEST_DIR = 'data/data_split/test'
EPOCHS = 5
CONTEXT = 7
BATCH_SIZE = 256

dtype = tf.float32
np_dtype = dtype.as_numpy_dtype
diff_coarse_to_id = load_id_dict("diff_coarse_to_id.txt")
audio_select_channels = '0,1,2'
channels = stride_csv_arg_list(audio_select_channels, 1, int)

# Loading data
def train_gen():
  song_paths = files_in(TRAIN_DIR)
  # TODO: Randomize file order
  # song_paths = ["data/train/Fraxtil_sArrowArrangements_PainGame.pkl", "data/train/Fraxtil_sArrowArrangements_NoBeginning.pkl"]
  return gen(song_paths)

def test_gen():
  song_paths = files_in(TEST_DIR)
  # song_paths = ["data/test/Fraxtil_sArrowArrangements_InnerUniverse_ExtendedMix_.pkl", "data/test2/Fraxtil_sArrowArrangements_Named_TheMoon_.pkl", "data/test2/Fraxtil_sBeastBeats_KokeshiNekoMedley.pkl", "data/test2/Fraxtil_sBeastBeats_Mess.pkl"]
  return gen(song_paths)

def gen(song_paths):
  for song in song_paths:
    song_data = None
    with open(song, 'rb') as f:
      song_data = reduce2np(pickle.load(f))
    # gc.collect()
    # print(f'Collected {gc.collect()} after pickle load')
    for chart_data in song_data:
      for time_step in chart_data:
        yield time_step
      # del chart_data
      # del time_step
    del song_data
    # gc.collect()
    # print(f'Collected {gc.collect()} after one song')

train_ds = tf.data.Dataset.from_generator(
    train_gen,
    (tf.float32, tf.uint8)
)
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_generator(
    test_gen,
    (tf.float32, tf.uint8)
)
test_ds = test_ds.batch(BATCH_SIZE) # This fixes mem issue, but shuffling should be used for trainset

# Model architecture
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(filters=10, kernel_size=(7,3), strides=(1,1), activation='relu')
    self.max1 = MaxPool2D(pool_size=(1,3), strides=(1,3))
    self.conv2 = Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), activation='relu')
    self.max2 = MaxPool2D(pool_size=(1,3), strides=(1,3))
    n_audiofeats = 8 * 20
    self.flatten = lambda x: tf.reshape(x, [x.shape[0], 7, n_audiofeats])
    self.rnn = RNN(StackedRNNCells([LSTMCell(units=512, dropout=0.5) for _ in range(2)]))
    self.fc1 = Dense(units=256, activation="relu")

  def call(self, x):
    x = self.conv1(x)
    x = self.max1(x)
    x = self.conv2(x)
    x = self.max2(x)
    # prtval('Shape after max2', x.shape)
    x = self.flatten(x)
    x = self.rnn(x)
    return self.fc1(x)

# Create an instance of the model
model = MyModel()

# Model compile settings
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Integrating these settings
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  return predictions


# Run the NN
if DO_TRAIN:
  print(f"Train and test for {EPOCHS} epochs")

  for epoch in range(EPOCHS):
    start = timer()
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
      train_step(np.asarray(images), np.asarray(labels))

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    end = timer()

    print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_loss.result()}, '
      f'Accuracy: {train_accuracy.result() * 100}, '
      f'Test Loss: {test_loss.result()}, '
      f'Test Accuracy: {test_accuracy.result() * 100}, '
      f'Time: {end - start}s'
    )
    model.save_weights(WEIGHTS_FILE+f'_epoch_{epoch}')
  model.save_weights(WEIGHTS_FILE)
  
else:
  print("Testing once")
  model.load_weights(WEIGHTS_FILE)
  start = timer()
  test_loss.reset_states()
  test_accuracy.reset_states()
  y_true = []
  y_pred = []

  for test_images, test_labels in test_ds:
    y_true.extend(test_labels)
    last_layer = test_step(test_images, test_labels)
    pred = np.argmax(last_layer, axis=1)
    y_pred.extend(pred)
  
  end = timer()

  print(
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}, '
    f'Time: {end - start}s'
  )

  confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
  plt.figure(figsize=(60, 48))
  sns.heatmap(confusion_mtx, annot=False, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()