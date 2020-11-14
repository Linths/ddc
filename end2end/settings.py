from enum import Enum, auto
import datetime

class RunMode(Enum):
  TRAIN_BASIC = auto()
  TEST_BASIC = auto()
  WITH_PRE_TRAIN = auto()

RUN_MODE = RunMode.WITH_PRE_TRAIN
BUILD_DATASET = False
BUILD_BALANCED_DATASET = True

timestamp = datetime.datetime.now().replace(microsecond=0).strftime('%Y-%m-%d %H_%M_%S')
WEIGHTS_FILE = f'weights/{timestamp}/weights'
PRE_WEIGHTS_FILE = f'pre-weights/{timestamp}/pre-weights'

DATA_DIR = '../data'
TRAIN_DIR = f'{DATA_DIR}/data_split/train'
TEST_DIR = f'{DATA_DIR}/data_split/test'
TF_DIR = f'{DATA_DIR}/tf'
TRAIN_TF_DIR = f'{TF_DIR}/train'
TEST_TF_DIR = f'{TF_DIR}/test'
BALANCED_TRAIN_TF_DIR = f'{TF_DIR}/balanced-train'

EPOCHS = 10
CONTEXT = 7
WINDOW = 2*CONTEXT+1
BATCH_SIZE = 256
N_CLASSES = 256

