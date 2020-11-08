import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
from tensorflow.keras import Model
from keras import backend as K
from tensorflow.python.framework import ops
# from keras.losses import LossFunctionWrapper
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

DO_TRAIN = True
SAVE_TF = False

WEIGHTS_FILE = 'weights/weights'
TRAIN_DIR = '../data/data_split/train'
TEST_DIR = '../data/data_split/test'
TF_DIR = '../data/tf'
TRAIN_TF_DIR = f'{TF_DIR}/train'
TEST_TF_DIR = f'{TF_DIR}/test'

EPOCHS = 100
CONTEXT = 7
WINDOW = 2*CONTEXT+1
BATCH_SIZE = 256

# Loading data
def song_gen(song_path):
  print(song_path)
  song_data = None
  with open(song_path, 'rb') as f:
    song_data = reduce2np(pickle.load(f))
  print(f'\t{len(song_data)} charts')
  for chart_data in song_data[:1]:
   print(f'\t{len(chart_data)} timesteps')
   for time_step in chart_data:
      yield time_step
  del song_data

def get_right_label(labels):
  return labels.skip(CONTEXT).take(1)

def build_song_ds(song_path):
  ds = tf.data.Dataset.from_generator(
    lambda: song_gen(song_path),
    (tf.float32, tf.uint8)
  )
  ds = ds.window(size=WINDOW, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(WINDOW), get_right_label(y))))
  return ds

def build_ds(dir):
  all_ds = [build_song_ds(file) for file in files_in(dir)]
  total_ds = all_ds[0]
  for ds in all_ds[1:]:
    total_ds = total_ds.concatenate(ds)
  i = 0
  for x,y in total_ds:
    i += 1
  print(f'[{i} total]')
  return total_ds

# def concat()

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
end = timer()
print(f'[Train] Loaded in {end-start} sec.')
print(len(train_ds_loaded))

start = timer()
test_ds_loaded = tf.data.experimental.load(TEST_TF_DIR,
    (tf.TensorSpec(shape=(WINDOW, 80, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(), dtype=tf.uint8)))
end = timer()
print(f'[Test] Loaded in {end-start} sec.')
print(len(test_ds_loaded))

pos_train_ds_loaded = train_ds_loaded.filter(lambda feats, label: label != 0)
neg_train_ds_loaded = train_ds_loaded.filter(lambda feats, label: label == 0)

def ds_len(ds):
  i = 0
  for _ in ds:
    i += 1
  return i

npos_train = ds_len(pos_train_ds_loaded)
nneg_train = ds_len(neg_train_ds_loaded)
rate_train = npos_train / (npos_train + nneg_train)
print(f'[Train] {rate_train * 100}% positive rate. {npos_train} vs {nneg_train}')
train_neg_shift = int(1/rate_train * 256)
print(train_neg_shift)
neg_train_ds_loaded = neg_train_ds_loaded.window(size=1, shift=train_neg_shift).flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(1), y.batch(1)))).unbatch()
print(f'[Train] Picked only {ds_len(neg_train_ds_loaded)} NO_STEPs, using shift={train_neg_shift}')

balanced_train = tf.data.experimental.sample_from_datasets([pos_train_ds_loaded, neg_train_ds_loaded], weights=[1/256, 255/256])
balanced_train = balanced_train.batch(BATCH_SIZE)
test_ds_loaded = test_ds_loaded.batch(BATCH_SIZE)

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
    x = self.flatten(x)
    x = self.rnn(x)
    return self.fc1(x)

# Create an instance of the model
model = MyModel()

# Model compile settings
# class_weights = [0.03] + 255*[0.97] # maybe everything /256
# ratio_hit = 0.0000000000001
# class_weights = [ratio_hit] + 255*[1-ratio_hit] # maybe everything /256
# class_weights = [.5] + 255*[.5 / 255]
# class_weights = 256*[1]

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
    # weighted_logits = predictions * class_weights
    # loss = loss_object(labels, weighted_logits)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  return predictions

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  # weighted_logits = predictions * class_weights
  # t_loss = loss_object(labels, weighted_logits)
  test_loss(t_loss)
  test_accuracy(labels, predictions)
  return predictions

def train(train_ds, test_ds):
  print(f"Train and test for {EPOCHS} epochs")

  for epoch in range(EPOCHS):
    start = timer()
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for feats, labels in train_ds:
      train_step(np.asarray(feats), np.asarray(labels))
    for feats, labels in test_ds:
      last_layer = test_step(feats, labels)
      preds = np.argmax(last_layer, axis=1)
      print(f'[test] steps@ {[i for i, step in enumerate(preds) if step == 0]}')
      print([num2step(pred) for pred in preds])

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

def test(test_ds, show_confmat=False):
  print("Testing once")
  model.load_weights(WEIGHTS_FILE)
  start = timer()
  test_loss.reset_states()
  test_accuracy.reset_states()
  y_true = []
  y_pred = []

  for feats, labels in test_ds:
    y_true.extend(labels)
    last_layer = test_step(feats, labels)
    # last_layer = last_layer[:, 1:] # FIXME: now never predicts NO_STEP
    pred = np.argmax(last_layer, axis=1)
    # pred = np.argmax(last_layer, axis=1) + 1 # FIXME: now never predicts NO_STEP
    y_pred.extend(pred)

  end = timer()

  print(
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}, '
    f'Time: {end - start}s'
  )

  if show_confmat:
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(60, 48))
    sns.heatmap(confusion_mtx, annot=False, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show() # TODO plt.savefig
  return y_pred

# Run the NN
if DO_TRAIN:
  train(balanced_train, test_ds_loaded)
  # train(train_ds_loaded, test_ds_loaded)
else:
  predicted_steps = test(test_ds_loaded)
  print([i for i, step in enumerate(predicted_steps) if step != 0])
