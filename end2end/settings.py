from enum import Enum, auto
import datetime

class RunMode(Enum):
  TRAIN_BASIC = auto()
  TEST_BASIC = auto()
  WITH_PRE_TRAIN = auto()
  TEST_ONLY_PRE = auto()
  TEST_FINETUNE = auto()

RUN_MODE = RunMode.TEST_FINETUNE #TEST_ONLY_PRE #WITH_PRE_TRAIN
BUILD_DATASET = False
BUILD_BALANCED_DATASET = False
BUILD_SPLIT_DATASET = False

timestamp = datetime.datetime.now().replace(microsecond=0).strftime('%Y-%m-%d %H_%M_%S')
WEIGHTS_FILE = f'weights/{timestamp}/weights'
PRE_WEIGHTS_FILE = f'pre-weights/{timestamp}/pre-weights'

DATA_DIR = '../data'
TRAIN_DIR = f'{DATA_DIR}/data_split/train'
TEST_DIR = f'{DATA_DIR}/data_split/test'
TF_DIR = f'{DATA_DIR}/tf'
TRAIN_TF_DIR = f'{TF_DIR}/train'
TEST_TF_DIR = f'{TF_DIR}/test'
ONE_TF_DIR = f'one'
BALANCED_TRAIN_TF_DIR = f'{TF_DIR}/balanced-train'
SPLIT_TRAIN_TF_DIR = f'{TF_DIR}/split-train'

EPOCHS = 1000
PRINT_EVERY_EPOCHS = 1
CONTEXT = 7
WINDOW = 2*CONTEXT+1
BATCH_SIZE = 256
N_CLASSES = 256

