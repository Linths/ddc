from enum import Enum, auto

class RunMode(Enum):
  TRAIN_BASIC = auto()
  TEST_BASIC = auto()
  WITH_PRE_TRAIN = auto()

RUN_MODE = RunMode.WITH_PRE_TRAIN
BUILD_DATASET = False

WEIGHTS_FILE = 'weights/weights'
PRE_WEIGHTS_FILE = 'pre-weights/pre-weights'

DATA_DIR = '../data'
TRAIN_DIR = f'{DATA_DIR}/data_split/train'
TEST_DIR = f'{DATA_DIR}/data_split/test'
TF_DIR = f'{DATA_DIR}/tf'
TRAIN_TF_DIR = f'{TF_DIR}/train'
TEST_TF_DIR = f'{TF_DIR}/test'

EPOCHS = 5
CONTEXT = 7
WINDOW = 2*CONTEXT+1
BATCH_SIZE = 256
N_CLASSES = 256