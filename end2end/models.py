# Model architecture
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
from tensorflow.keras import Model
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from util2 import num2step
from settings import N_CLASSES, PRINT_EVERY_EPOCHS

class ChoreographModel(Model):
  def __init__(self):
    super(ChoreographModel, self).__init__()
    self.conv1 = Conv2D(filters=10, kernel_size=(7,3), strides=(1,1), kernel_initializer=GlorotUniform(seed=1), activation='relu', name='conv1')
    self.max1 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max1')
    self.conv2 = Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), kernel_initializer=GlorotUniform(seed=1), activation='relu', name='conv2')
    self.max2 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max2')
    n_audiofeats = 8 * 20
    self.flatten = lambda x: tf.reshape(x, [x.shape[0], 7, n_audiofeats])
    self.rnn = RNN(StackedRNNCells([LSTMCell(units=512, dropout=0.5) for _ in range(2)]), name='rnn')
    self.fc = Dense(units=N_CLASSES, activation="relu", name='final')

  def call(self, x):
    x = self.conv1(x)
    x = self.max1(x)
    x = self.conv2(x)
    x = self.max2(x)
    x = self.flatten(x)
    x = self.rnn(x)
    return self.fc(x)

class PreTrainModel(Model):
  def __init__(self):
    super(PreTrainModel, self).__init__()
    self.conv1 = Conv2D(filters=10, kernel_size=(7,3), strides=(1,1), kernel_initializer=GlorotUniform(seed=1), activation='relu', name='conv1')
    self.max1 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max1')
    self.conv2 = Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), kernel_initializer=GlorotUniform(seed=1), activation='relu', name='conv2')
    self.max2 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max2')
    self.pre_flatten = Flatten(name='pre_flatten')
    self.pre_fc = Dense(units=N_CLASSES, activation="relu", name='pre-final')

  def call(self, x):
    x = self.conv1(x)
    x = self.max1(x)
    x = self.conv2(x)
    x = self.max2(x)
    x = self.pre_flatten(x)
    return self.pre_fc(x)


def run_step(feats, labels, model,
            is_train, is_weighted, is_finetuning, 
            loss_object, loss_metrics, accuracy_metrics,
            class_weights=None, optimizer=None,
            optimizer_pretrained_layers=None):

  # def predict_step():
  #   predictions = model(feats, training=is_train)
  #   if not is_weighted:
  #     loss = loss_object(labels, predictions)
  #   else:
  #     assert class_weights != None
  #     weighted_logits = predictions * class_weights
  #     loss = loss_object(labels, weighted_logits)
  #   return predictions, loss

  if is_train:
    assert optimizer != None
    with tf.GradientTape() as tape:
      # predictions, loss = predict_step()
      predictions = model(feats, training=is_train)
      if not is_weighted:
        loss = loss_object(labels, predictions)
      else:
        assert class_weights != None
        weighted_logits = predictions * class_weights
        loss = loss_object(labels, weighted_logits)
    variables = model.trainable_variables

    if not is_finetuning:
      gradients = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))
    else:
      assert optimizer_pretrained_layers != None
      n = 4 # n_pretrained_layers
      pretrain_vars = variables[-n:]
      other_vars = variables[:-n]

      gradients = tape.gradient(loss, other_vars + pretrain_vars)
      pretrain_grads = gradients[-n:] # cnn layers gradients
      other_grads = gradients[:-n] # other gradients
      
      optimizer_pretrained_layers.apply_gradients(zip(pretrain_grads, pretrain_vars))
      optimizer.apply_gradients(zip(other_grads, other_vars))
  
  else:
    # predictions, loss = predict_step()
    predictions = model(feats, training=is_train)
    if not is_weighted:
      loss = loss_object(labels, predictions)
    else:
      assert class_weights != None
      weighted_logits = predictions * class_weights
      loss = loss_object(labels, weighted_logits)
  
  loss_metrics(loss)
  accuracy_metrics(labels, predictions)
  return predictions


def load_pretrained_weights(pretrain_model, finetune_model, preweights_file):
  pretrain_model.load_weights(preweights_file)
  finetune_model.conv1 = pretrain_model.conv1
  finetune_model.max1 = pretrain_model.max1
  finetune_model.conv2 = pretrain_model.conv2
  finetune_model.max2 = pretrain_model.max2
  assert finetune_model.variables != []


def run_total(model, test_ds, test_step_fn,
              epochs, metrics, weights_file=None,
              train_ds=None, train_step_fn=None,
              show_confmat=False):
  print(f"Train and test for {epochs} epochs")

  if weights_file != None:
    subprocess.run(["cp", "settings.py", weights_file])

  for epoch in range(epochs):
    start = timer()
    for metric in metrics:
      metric.reset_states()
    y_true = []
    y_pred = []
    
    if train_ds != None and train_step_fn != None:
      for feats, labels in train_ds:
        train_step_fn(feats, labels, model)
    
    for feats, labels in test_ds:
      last_layer = test_step_fn(feats, labels, model)
      preds = np.argmax(last_layer, axis=1)
      if epoch % PRINT_EVERY_EPOCHS == 0:
        _print_prediction_summary(preds)
      if show_confmat:
        y_true.extend(labels)
        y_pred.extend(preds)

    end = timer()
    _print_epoch_summary(epoch, metrics, start, end)

    if weights_file != None:
      model.save_weights(weights_file + _epoch_file_postfix(epoch, metrics, start, end))
  if weights_file != None:
    model.save_weights(weights_file)
  
  if show_confmat:
    _show_confmat(y_true, y_pred)
  
  return y_pred


def _show_confmat(y_true, y_pred):
  confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(60, 48))
  sns.heatmap(confusion_mtx, annot=False, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show() # TODO plt.savefig

def _print_epoch_summary(epoch, metrics, start, end):
  text = f'Epoch {epoch + 1}, '
  for metric in metrics:
    text += f'{metric.name}: {metric.result():.3f}, '
  text += f'time: {(end-start):.3f}s'
  print(text)

def _epoch_file_postfix(epoch, metrics, start, end):
  text = f' epoch {epoch + 1}, '
  for metric in metrics:
    text += f'{metric.name} {metric.result():.3f}, '
  text += f'time {(end-start):.3f}s'
  return text

def _print_prediction_summary(preds):
  print(f'0s @ {[i for i, step in enumerate(preds) if step == 0]}')
  print(f'xs @ {[i for i, step in enumerate(preds) if step != 0]}')
  print([num2step(pred) for pred in preds])
