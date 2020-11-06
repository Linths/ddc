import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
from tensorflow.keras import Model
import numpy as np
import pickle
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
import subprocess

from util import *
from util2 import *

np.set_printoptions(precision=3)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

DO_TRAIN = False
SAVE_TF = False

WEIGHTS_FILE = 'weights/weights'
TRAIN_DIR = '../data/data_split/train'
TEST_DIR = '../data/data_split/test'
TF_DIR = '../data/tf'
TRAIN_TF_DIR = f'{TF_DIR}/train'
TEST_TF_DIR = f'{TF_DIR}/test'

EPOCHS = 5
CONTEXT = 7
WINDOW = 2*CONTEXT+1
BATCH_SIZE = 256

# Loading data
def song_gen(song_path):
  print(song_path)
  song_data = None
  with open(song_path, 'rb') as f:
    song_data = reduce2np(pickle.load(f))
  for chart_data in song_data:
    for time_step in chart_data:
      yield time_step
  del song_data

def get_right_label(labels):
  return labels.skip(CONTEXT).take(1)

def build_ds(dir):
  total_ds = None
  for song_path in files_in(dir):
    #print(song_path)
    ds = tf.data.Dataset.from_generator(
      lambda: song_gen(song_path),
      (tf.float32, tf.uint8)
    )
    ds = ds.window(size=WINDOW, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(WINDOW), get_right_label(y))))
    for x,y in ds:
      pass

    if total_ds == None:
      total_ds = ds
    else:
      total_ds.concatenate(ds)
  return total_ds

if SAVE_TF:
  subprocess.run(["rm", "-rf", f"{TRAIN_TF_DIR}/*"])
  subprocess.run(["rm", "-rf", f"{TEST_TF_DIR}/*"])

  start = timer()
  train_ds = build_ds(TRAIN_DIR)
  tf.data.experimental.save(train_ds, TRAIN_TF_DIR)
  end = timer()
  print(f'[Train] Done in {end-start} sec.')

  start = timer()
  test_ds = build_ds(TEST_DIR)
  tf.data.experimental.save(test_ds, TEST_TF_DIR)
  end = timer()
  print(f'[Test] Done in {end-start} sec.')

  subprocess.run(["du", "-h", TF_DIR])


start = timer()
train_ds_loaded = tf.data.experimental.load(TRAIN_TF_DIR,
    (tf.TensorSpec(shape=(WINDOW, 80, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(), dtype=tf.uint8)))
train_ds_loaded = train_ds_loaded.batch(BATCH_SIZE)
end = timer()
print(f'[Train] Batched in {end-start} sec.')
print(len(train_ds_loaded))

start = timer()
test_ds_loaded = tf.data.experimental.load(TEST_TF_DIR,
    (tf.TensorSpec(shape=(WINDOW, 80, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(), dtype=tf.uint8)))
test_ds_loaded = test_ds_loaded.batch(BATCH_SIZE)
end = timer()
print(f'[Test] Batched in {end-start} sec.')
print(len(test_ds_loaded))

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

    for feats, labels in train_ds_loaded:
      train_step(np.asarray(feats), np.asarray(labels))
    for feats, labels in test_ds_loaded:
      test_step(feats, labels)

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

  for test_images, test_labels in test_ds_loaded:
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
