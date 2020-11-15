import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
import numpy as np
import pickle
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
import subprocess
import math

from util import *
from util2 import num2step
from models import ChoreographModel, PreTrainModel, run_step, run_total, load_pretrained_weights
from settings import RunMode, RUN_MODE, BUILD_DATASET, BUILD_BALANCED_DATASET, WEIGHTS_FILE, PRE_WEIGHTS_FILE, DATA_DIR, TRAIN_DIR, TEST_DIR, TF_DIR, TRAIN_TF_DIR, TEST_TF_DIR, ONE_TF_DIR, BALANCED_TRAIN_TF_DIR, EPOCHS, CONTEXT, WINDOW, BATCH_SIZE, N_CLASSES
from data_loader import build_dataset, load_dataset, undersample, stratified_sample
from chart_writer import write

np.set_printoptions(precision=3)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if BUILD_DATASET:
  build_dataset(load_dir=TRAIN_DIR, save_dir=TRAIN_TF_DIR, name='train')
  build_dataset(load_dir=TEST_DIR, save_dir=TEST_TF_DIR, name='test')
  subprocess.run(["du", "-h", TF_DIR])

train_ds = load_dataset(TRAIN_TF_DIR, name='train')
test_ds = load_dataset(TEST_TF_DIR, name='test')
one_ds = load_dataset(ONE_TF_DIR, name='one')

if BUILD_BALANCED_DATASET:
  start = timer()
  balanced_train_ds = stratified_sample(train_ds, class_size=15000)
  tf.data.experimental.save(balanced_train_ds, BALANCED_TRAIN_TF_DIR)
  end = timer()
  print(f'Balancing train dataset took {end-start}s')
  subprocess.run(["du", "-h", BALANCED_TRAIN_TF_DIR])
else:
  balanced_train_ds = load_dataset(BALANCED_TRAIN_TF_DIR, name='balanced-train')

train_ds = train_ds.batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)
one_ds = one_ds.batch(BATCH_SIZE)
balanced_train_ds = balanced_train_ds.batch(BATCH_SIZE)


# === MODEL COMPILE SETTINGS ===
ratio_empty = 0.985
mu = 3 #0.15
class_weights = [1/ratio_empty] + (N_CLASSES-1)*[1/((1-ratio_empty)/(N_CLASSES-1))]
#class_weights = [math.log(mu*x) for x in class_weights]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

pre_lr = 0.001
cnn_lr = 0.000001
res_lr = 0.0001
pre_optimizer = tf.keras.optimizers.Adam(learning_rate=pre_lr)
cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_lr)
res_optimizer = tf.keras.optimizers.Adam(learning_rate=res_lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
all_metrics = [train_loss, train_accuracy, test_loss, test_accuracy]
test_metrics = [test_accuracy]

train_kwargs = {'is_train': True, 'loss_object': loss_object,
  'loss_metrics': train_loss, 'accuracy_metrics': train_accuracy,
  'class_weights': class_weights}
test_kwargs = {'is_train': False, 'is_weighted': False, 'loss_object': loss_object,
  'loss_metrics': test_loss, 'accuracy_metrics': test_accuracy}

if RUN_MODE == RunMode.WITH_PRE_TRAIN:
  pre_model = PreTrainModel()
  mock_ds = [(np.zeros((BATCH_SIZE,WINDOW,80,3)), np.zeros(BATCH_SIZE))]
 
  @tf.function
  def pretrain_step(x,y,m):
      return run_step(x, y, m,
            is_weighted=False,
            is_finetuning=False,
            optimizer=pre_optimizer,
            **train_kwargs)
  @tf.function
  def pretest_step(x,y,m):
    return run_step(x, y, m,
            is_finetuning=False,
            **test_kwargs)
  run_total(pre_model,
            train_ds=mock_ds, #balanced_train_ds,
            test_ds=mock_ds, #test_ds,
            train_step_fn=pretrain_step,
            test_step_fn=pretest_step,
            epochs=1, #EPOCHS,
            metrics=all_metrics,
            weights_file=None) #PRE_WEIGHTS_FILE)

  model = ChoreographModel()
  load_pretrained_weights(
            pretrain_model=pre_model,
            finetune_model=model,
            preweights_file="pre-weights/2020-11-15 13_08_29/pre-weights epoch 40, train_loss 2.059, train_accuracy 0.635, test_loss 5.609, test_accuracy 0.845, time 127.518s")
  @tf.function
  def finetrain_step(x,y,m):
    return run_step(x, y, m,
            is_weighted=False,
            is_finetuning=True,
            optimizer=res_optimizer,
            optimizer_pretrained_layers=cnn_optimizer,
            **train_kwargs)
  @tf.function
  def finetest_step(x,y,m):
    return run_step(x, y, m,
            is_finetuning=True,
            **test_kwargs)
  run_total(model,
            train_ds=balanced_train_ds, #train_ds,
            test_ds=test_ds,
            train_step_fn=finetrain_step,
            test_step_fn=finetest_step,
            epochs=EPOCHS,
            metrics=all_metrics,
            weights_file=WEIGHTS_FILE)

elif RUN_MODE == RunMode.TEST_ONLY_PRE:
  pre_model = PreTrainModel()
  mock_ds = [(np.zeros((BATCH_SIZE,WINDOW,80,3)), np.zeros(BATCH_SIZE))]
 
  @tf.function
  def pretrain_step(x,y,m):
      return run_step(x, y, m,
            is_weighted=False,
            is_finetuning=False,
            optimizer=pre_optimizer,
            **train_kwargs)
  @tf.function
  def pretest_step(x,y,m):
    return run_step(x, y, m,
            is_finetuning=False,
            **test_kwargs)
  run_total(pre_model,
            train_ds=mock_ds, #balanced_train_ds,
            test_ds=mock_ds, #test_ds,
            train_step_fn=pretrain_step,
            test_step_fn=pretest_step,
            epochs=1, #EPOCHS,
            metrics=all_metrics,
            weights_file=None) #PRE_WEIGHTS_FILE)
  pre_model.load_weights("pre-weights/2020-11-15 13_08_29/pre-weights epoch 40, train_loss 2.059, train_accuracy 0.635, test_loss 5.609, test_accuracy 0.845, time 127.518s")
  assert pre_model.variables != []

  predictions = run_total(pre_model,
            train_ds=None,
            test_ds=one_ds,
            train_step_fn=None,
            test_step_fn=pretest_step,
            epochs=1,
            metrics=all_metrics,
            weights_file=None,
            ret_preds=True)
  print(f'Predictions len is {len(predictions)}')
  write(predictions)

elif RUN_MODE == RunMode.TRAIN_BASIC:
  model = ChoreographModel()
  @tf.function
  def train_step(x,y,m):
    return run_step(x, y, m,
            is_train=True,
            is_weighted=False,
            is_finetuning=False,
            optimizer=res_optimizer,
            **train_kwargs)
  @tf.function
  def test_step(x,y,m):
    return run_step(x, y, m,
            is_train=False,
            is_weighted=False,
            is_finetuning=False,
            **test_kwargs)
  run_total(model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_step_fn=train_step,
            test_step_fn=test_step,
            epochs=EPOCHS,
            metrics=all_metrics,
            weights_file=WEIGHTS_FILE)

elif RUN_MODE == RunMode.TEST_BASIC:
  model = ChoreographModel()
  @tf.function
  def test_step(x,y,m):
    return run_step(x, y, m,
            is_train=False,
            is_weighted=False,
            is_finetuning=False,
            **test_kwargs)
  run_total(model,
            test_ds=test_ds,
            test_step_fn=test_step,
            epochs=1,
            metrics=test_metrics)

else:
  print('Haven\'t selected a run mode.')
