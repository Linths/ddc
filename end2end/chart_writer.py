from essentia.standard import MetadataReader
from extract_feats import extract_mel_feats
from extract_feats import create_analyzers
import uuid
import zipfile
import shutil
import os
import pickle

from util2 import num2step, EMPTY_STEP

_DIFFS = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
_DIFF_TO_COARSE_FINE_AND_THRESHOLD = {
    'Beginner':     (0, 1, 0.15325437),
    'Easy':         (1, 3, 0.23268291),
    'Medium':       (2, 5, 0.29456162),
    'Hard':         (3, 7, 0.29084727),
    'Challenge':    (4, 9, 0.28875697)
}

_SUBDIV = 192
_DT = 0.01
_HZ = 1.0 / _DT
_BPM = 60 * (1.0 / _DT) * (1.0 / float(_SUBDIV)) * 4.0

_TEMPL = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:0.0;
#BPMS:0.0={bpm};
#STOPS:;
{charts}\
"""

_PACK_NAME = 'DDR_on_Demand_v0'
_CHART_TEMPL = """\
#NOTES:
    dance-single:
    DDR_on_Demand_v0:
    {ccoarse}:
    {cfine}:
    0.0,0.0,0.0,0.0,0.0:
{measures};\
"""

class CreateChartException(Exception):
  pass

def create_chart_dir(
    artist, title,
    audio_fp,
    norm, analyzers,
    diffs, idx_to_label,
    labels, out_dir, delete_audio=False):
  if not artist or not title:
    print('Extracting metadata from {}'.format(audio_fp))
    meta_reader = MetadataReader(filename=audio_fp)
    metadata = meta_reader()
    if not artist:
      artist = metadata[1]
    if not artist:
      artist = 'Unknown Artist'
    if not title:
      title = metadata[0]
    if not title:
      title = 'Unknown Title'

  print('Loading {} - {}'.format(artist, title))
  try:
    song_feats = extract_mel_feats(audio_fp, analyzers, nhop=441)
  except:
    raise CreateChartException('Invalid audio file: {}'.format(audio_fp))
  song_feats -= norm[0]
  song_feats /= norm[1]
  song_len_sec = song_feats.shape[0] / _HZ
  print('Processed {} minutes of features'.format(song_len_sec / 60.0))

  diff_chart_txts = []
  for diff in diffs:
    try:
      coarse, fine, threshold = _DIFF_TO_COARSE_FINE_AND_THRESHOLD[diff]
    except KeyError:
      raise CreateChartException('Invalid difficulty: {}'.format(diff))

    # TODO: Convert audio to feats & do prediction magic
    # FIXME: Hardcoded test_ds_loaded, instead of extracting feats from audio

    predicted_steps = [num2step(x) for x in labels] #test(test_ds_loaded)]
    
    print('Creating chart text')
    time_to_step = {t : step for t, step in enumerate(predicted_steps)}
    max_subdiv = max(time_to_step.keys())
    if max_subdiv % _SUBDIV != 0:
      max_subdiv += _SUBDIV - (max_subdiv % _SUBDIV)
    full_steps = [time_to_step.get(i, EMPTY_STEP) for i in range(max_subdiv)]
    measures = [full_steps[i:i+_SUBDIV] for i in range(0, max_subdiv, _SUBDIV)]
    measures_txt = '\n,\n'.join(['\n'.join(str(measure)) for measure in measures])
    chart_txt = _CHART_TEMPL.format(
      ccoarse=_DIFFS[coarse],
      cfine=fine,
      measures=measures_txt
    )
    diff_chart_txts.append(chart_txt)

  print('Creating SM')
  out_dir_name = os.path.split(out_dir)[1]
  audio_out_name = out_dir_name + os.path.splitext(audio_fp)[1]
  sm_txt = _TEMPL.format(
    title=title,
    artist=artist,
    music_fp=audio_out_name,
    bpm=_BPM,
    charts='\n'.join(diff_chart_txts))

  print('Saving to {}'.format(out_dir))
  try:
    os.mkdir(out_dir)
    audio_ext = os.path.splitext(audio_fp)[1]
    shutil.copyfile(audio_fp, os.path.join(out_dir, audio_out_name))
    with open(os.path.join(out_dir, out_dir_name + '.sm'), 'w') as f:
      f.write(sm_txt)
  except:
    raise CreateChartException('Error during output')

  if delete_audio:
    try:
      os.remove(audio_fp)
    except:
      raise CreateChartException('Error deleting audio')

  return True

def create_chart_closure(artist, title, audio_fp, norm, analyzers, diffs, idx_to_label, labels, out_dir):
  song_id = uuid.uuid4()
  out_dir = os.path.join(out_dir, str(song_id))
  try:
    create_chart_dir(
      artist=artist,
      title=title,
      audio_fp=audio_fp,
      norm=norm,
      analyzers=analyzers,
      diffs=diffs,
      idx_to_label=idx_to_label,
      labels=labels,
      out_dir=out_dir)
  except CreateChartException as e:
    raise e
  except Exception as e:
    raise CreateChartException('Unknown chart creation exception')
  return

def write(labels):
  print('Loading band norms')
  with open('write_settings/norm.pkl', 'rb') as f:
    norm = pickle.load(f, encoding = 'bytes')

  print('Creating Mel analyzers')
  analyzers = create_analyzers(nhop=441)

  print('Loading labels')
  with open('write_settings/labels_4_0123.txt', 'r') as f:
    idx_to_label = {i + 1:l for i, l in enumerate(f.read().splitlines())}

  # Perhaps some preloading of the model

  artist = 'Fraxtil'
  title = 'Mess' #Inner Universe (Extended Mix)'
  audio_fp = 'Mess.ogg' # 'Inner Universe (Extended Mix).ogg' #'Black_Magic.ogg'
  diffs = ['Easy'] #['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
  out_dir = "output"
  create_chart_closure(artist=artist, title=title, audio_fp=audio_fp, norm=norm, analyzers=analyzers, diffs=diffs, idx_to_label=idx_to_label, labels=labels, out_dir=out_dir)

