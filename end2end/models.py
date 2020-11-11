# Model architecture
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPool2D, MaxPool3D, LSTM, LSTMCell, StackedRNNCells, RNN
from tensorflow.keras import Model

N_CLASSES = 256

class ChoreographModel(Model):
  def __init__(self):
    super(ChoreographModel, self).__init__()
    self.conv1 = Conv2D(filters=10, kernel_size=(7,3), strides=(1,1), activation='relu', name='conv1')
    self.max1 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max1')
    self.conv2 = Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), activation='relu', name='conv2')
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
    self.conv1 = Conv2D(filters=10, kernel_size=(7,3), strides=(1,1), activation='relu', name='conv1')
    self.max1 = MaxPool2D(pool_size=(1,3), strides=(1,3), name='max1')
    self.conv2 = Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), activation='relu', name='conv2')
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
