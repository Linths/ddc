import tensorflow as tf
import pickle
from timeit import default_timer as timer
import subprocess
import numpy as np

from util import *
from util2 import reduce2np, files_in, ds_len, prefix_print, num2step
from settings import CONTEXT, WINDOW, N_CLASSES, BATCH_SIZE

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
  class_dss = [ds.filter(lambda feats, label: label == i) for i in range(N_CLASSES)]
  print(class_dss)
  class_counts = [ds_len(ds) for ds in class_dss]
  print(class_counts)
  print([(num2step(x),y) for x,y in enumerate(class_counts) if y != 0])
  return class_dss, class_counts

def stratified_sample(input_ds, class_size=200):
  start = timer()
  class_dss, class_counts = _split_classes(input_ds)
  end = timer()
  print(f'Class splitting took {(end-start):.3f}s')

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
    if class_counts[i] == 0:
      print('Skip repeating due to no samples')
      balanced_class_ds = ds
    else:
      balanced_class_ds = ds.repeat().take(class_size)
      # print(ds_len(balanced_class_ds)) # Commented to save redundant iterations
    end = timer()
    print(f'-> took {(end-start):.3f}s')
    balanced_class_dss.append(balanced_class_ds)
  balanced_ds = tf.data.experimental.sample_from_datasets(balanced_class_dss, seed=1)
  return balanced_ds