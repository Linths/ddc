import tensorflow as tf
import pickle
from timeit import default_timer as timer
import subprocess
import numpy as np

from util import *
from util2 import reduce2np, files_in, ds_len, prefix_print, num2step
from settings import CONTEXT, WINDOW, N_CLASSES, BATCH_SIZE, SPLIT_TRAIN_TF_DIR, BUILD_SPLIT_DATASET

# Loading data
def __song_gen(song_path):
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

def __get_right_label(labels):
  return labels.skip(CONTEXT).take(1)

def __build_song_ds(song_path):
  ds = tf.data.Dataset.from_generator(
    lambda: __song_gen(song_path),
    (tf.float32, tf.uint8)
  )
  ds = ds.window(size=WINDOW, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(WINDOW), __get_right_label(y))))
  return ds

def __build_ds(dir):
  all_ds = [__build_song_ds(file) for file in files_in(dir)]
  total_ds = all_ds[0]
  for ds in all_ds[1:]:
    total_ds = total_ds.concatenate(ds)
  i = 0
  for x,y in total_ds:
    i += 1
  print(f'[{i} total]')
  return total_ds

def build_dataset(load_dir, save_dir, name=None):
  subprocess.run(["rm", "-rf", f"{save_dir}/*"])

  start = timer()
  ds = __build_ds(load_dir)
  tf.data.experimental.save(ds, save_dir)
  end = timer()
  print(f'{prefix_print(name)}Done in {end-start} sec.')
  return ds  

def load_dataset(dir, name=None):
  start = timer()
  ds = tf.data.experimental.load(dir,
      (tf.TensorSpec(shape=(WINDOW, 80, 3), dtype=tf.float32),
      tf.TensorSpec(shape=(), dtype=tf.uint8)))
  end = timer()
  print(f'{prefix_print(name)}Loaded {len(ds)} in {end-start} sec.')
  return ds

def undersample(ds):
  pos_ds = ds.filter(lambda feats, label: label != 0)
  neg_ds = ds.filter(lambda feats, label: label == 0)

  n_pos = ds_len(pos_ds)
  n_neg = ds_len(neg_ds)
  rate_pos = n_pos / (n_pos + n_neg)
  print(f'[Undersample] {rate_pos * 100}% positive rate. {n_pos} vs {n_neg}')
  n_neg_allowed = n_pos/(N_CLASSES-1)
  neg_shift = int(n_neg / n_neg_allowed)
  neg_ds = neg_ds.window(size=1, shift=neg_shift).flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(1), y.batch(1)))).unbatch()
  print(f'[Undersample] Picked only {ds_len(neg_ds)} NO_STEPs out of {n_neg}, using shift={neg_shift}')

  balanced_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[(N_CLASSES-1)/N_CLASSES, 1/N_CLASSES], seed=1)
  balanced_ds = balanced_ds.batch(BATCH_SIZE)
  return balanced_ds

def oversample(ds):
  pos_ds = ds.filter(lambda feats, label: label != 0)
  neg_ds = ds.filter(lambda feats, label: label == 0)

  n_pos = ds_len(pos_ds)
  n_neg = ds_len(neg_ds)
  rate_pos = n_pos / (n_pos + n_neg)
  print(f'[Undersample] {rate_pos * 100}% positive rate. {n_pos} vs {n_neg}')
  n_pos_desired = n_neg * (N_CLASSES-1)
  print(n_pos_desired / n_pos)
  repeat_count = int(n_pos_desired / n_pos)
  print(f'{n_pos_desired} pos desired, repeating pos ({n_pos}) {repeat_count} times')
  pos_ds = pos_ds.repeat(repeat_count)

  balanced_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[(N_CLASSES-1)/N_CLASSES, 1/N_CLASSES], seed=1)
  balanced_ds = balanced_ds.batch(BATCH_SIZE)
  return balanced_ds

def _split_classes(ds):
  # Hardcoding class counts already calculated to save time
  #class_counts = []
  class_counts = [943561, 3602, 82, 45, 3793, 30, 0, 10, 74, 1, 0, 9, 38, 7, 10, 0, 3730, 36, 0, 13, 101, 0, 0, 0, 1, 0, 0, 0, 24, 0, 0, 0, 65, 0, 0, 12, 1, 0, 0, 0, 4, 0, 0, 0, 23, 0, 0, 0, 26, 11, 11, 0, 18, 0, 0, 0, 29, 0, 0, 0, 4, 0, 0, 0, 3608, 216, 2, 19, 35, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 25, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 2, 23, 11, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 9, 14, 23, 10, 0, 0, 0, 9, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
  if BUILD_SPLIT_DATASET:
    start = timer()
    class_dss = [ds.filter(lambda feats, label: label == i) for i in range(N_CLASSES)]
    end = timer()
    print(f'Filtering each label took {(end-start):.3f}s')
    
    start_all = timer()
    for i, class_ds in enumerate(class_dss):
      start = timer()
      print(f'Count for label {i}')
      #count = ds_len(class_ds)
      #class_counts.append(count)
      count = class_counts[i]
      if count != 0:
        print(f'\tSave dataset of size {count}')
        tf.data.experimental.save(class_ds, f'{SPLIT_TRAIN_TF_DIR}/{i}')
      else:
        print('\tSkip saving because 0 count')
      end = timer()
      print(f'\tCounting & possibly saving took {(end-start):.3f}s')
    end_all = timer()
    print(f'Total counting & saving took {(end_all-start_all):.3f}s')

  else:
    class_dss = []
    for i, count in enumerate(class_counts):
      start = timer()
      if count != 0:
        print(f'\tLoad dataset of size {count}')
        class_ds = load_dataset(f'{SPLIT_TRAIN_TF_DIR}/{i}', name=f'class_{i}')
        class_dss.append(class_ds)
      else:
        print('\tSkip loading because 0 count')
        class_ds.append(None)
      end = timer()
      print(f'\tLoading saving took {(end-start):.3f}s')

  print(class_counts)
  print([(num2step(x),y) for x,y in enumerate(class_counts) if y != 0])
  return class_dss, class_counts

def stratified_sample(input_ds, class_size=200):
  start = timer()
  class_dss, class_counts = _split_classes(input_ds)
  end = timer()
  print(f'In total, class splitting took {(end-start):.3f}s')

  n_empty = class_counts[0]
  other_counts = class_counts[1:]
  n_other = sum(other_counts)
  print(other_counts)
  popular_other = np.argmax(other_counts) + 1 
  n_popular_other = max(other_counts)
  # class_size = n_popular_other
  print(f'{n_empty} empties, {n_other} others')
  print(f'\"{num2step(popular_other)}\" is the most popular non-empty, appears {n_popular_other} times.')

  print(f'Converting every class to have size {class_size}')
  balanced_class_dss = []
  for i,ds in enumerate(class_dss):
    print(f'Label {i} with original count {class_counts[i]}')
    start = timer()
    if ds == None or class_counts[i] == 0:
      print('Skip dataset due to no samples')
    else:
      balanced_class_ds = ds.repeat().take(class_size)
      balanced_class_dss.append(balanced_class_ds)
      # print(ds_len(balanced_class_ds)) # Commented to save redundant iterations
    end = timer()
    print(f'-> took {(end-start):.3f}s')
  print(len(balanced_class_dss))
  print('Start with making balanced_ds')
  balanced_ds = tf.data.experimental.sample_from_datasets(balanced_class_dss, seed=1)
  print('Done with making balanced_ds')
  return balanced_ds
