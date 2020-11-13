import tensorflow as tf
import pickle
from timeit import default_timer as timer
import subprocess

from util import *
from util2 import reduce2np, files_in, ds_len, prefix_print
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
  print(f'{prefix_print(name)}Loaded in {end-start} sec.')
  print(len(ds))
  return ds

def undersample(ds):
  pos_ds = ds.filter(lambda feats, label: label != 0)
  neg_ds = ds.filter(lambda feats, label: label == 0)

  n_pos = ds_len(pos_ds)
  n_neg = ds_len(neg_ds)
  rate_pos = n_pos / (n_pos + n_neg)
  print(f'[Train] {rate_pos * 100}% positive rate. {n_pos} vs {n_neg}')
  neg_shift = int(1/rate_pos * N_CLASSES)
  print(neg_shift)
  neg_ds = neg_ds.window(size=1, shift=neg_shift).flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(1), y.batch(1)))).unbatch()
  print(f'[Train] Picked only {ds_len(neg_ds)} NO_STEPs, using shift={neg_shift}')

  balanced_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[1/N_CLASSES, (N_CLASSES-1)/N_CLASSES])
  balanced_ds = balanced_ds.batch(BATCH_SIZE)
  return balanced_ds