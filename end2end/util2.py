from os import listdir
from os.path import isfile, join
import numpy as np

np.set_printoptions(precision=3)

def prtval(string, value):
  print(f"{string:19s}: {value}")

def files_in(dir_path):
  if not dir_path.endswith('/'):
    dir_path += '/'
  return [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]

def step2num(step_string):
  # Ex. converts a 3002 step to the number 194 = 0b11000010
  result = 0
  for i in range(4):
    tile_state = int(step_string[i])
    bitshift = (3-i)*2
    result += tile_state << bitshift
  return result
empty_step = step2num('0000')

# Tests
assert step2num('3002') == 194
assert step2num('3003') == 195
correct_num = 0
for i in range(4):
  for j in range(4):
    for k in range(4):
      for l in range(4):
        num = step2num(str(i)+str(j)+str(k)+str(l))
        assert num == correct_num
        correct_num += 1

def get_context_data(chart):
  nframes = chart.get_nframes()
  onsets_steps = {int(round(t / chart.dt)): step2num(step) for _, _, t, step in chart.annotations}
  result = [get_context_data_at(chart, i, onsets_steps) for i in range(CONTEXT, nframes-CONTEXT)]
  return result

def get_context_data_at(chart, index, onsets_steps):
  feats, diff, _ = chart.get_example(frame_idx=index, dtype=np_dtype, time_context_radius=CONTEXT, diff_coarse_to_id=diff_coarse_to_id)
  label = onsets_steps.get(index, empty_step)
  return (feats, label)

def reduce2np(data):
  select_channels([data], channels)
  charts = flatten_dataset_to_charts([data])
  return [get_context_data(chart) for chart in charts]
