import os

from ddc import BeatTimeCalc


class Song(object):
  def __init__(self, file_dir, attrs):
    self.ezid = attrs['ezid']
    self.ezpack = attrs['ezpack']
    self.eztitle = attrs['ezname']

    self.artist = attrs['artist']
    self.title = attrs['title']

    self.btcalc = BeatTimeCalc(attrs['offset'], attrs['bpms'], attrs['stops'])

    self.audio_fp = os.path.join(file_dir, attrs['audio_fp'])

    self.charts_raw = attrs['charts']
    self.charts = None


  def get_audio_fp(self):
    return self.audio_fp


  def get_placement_charts(self):
    if self.charts is None:
      self.charts = []
      for chart_attrs in self.charts_raw:
        self.charts.append(PlacementChart(self, chart_attrs))
    return self.charts


class Chart(object):
  def __init__(self, song, chart_attrs):
    self.song = song

    self.type = chart_attrs['type']
    self.stepper = chart_attrs['stepper']
    self.difficulty = chart_attrs['difficulty']
    self.difficulty_fine = chart_attrs['difficulty_fine']

    self.steps = chart_attrs['steps']

  def get_difficulty(self):
    return self.difficulty


class PlacementChart(Chart):
  def __init__(self, song, chart_attrs):
    super(PlacementChart, self).__init__(song, chart_attrs)


  def get_audio_fp(self):
    return self.song.get_audio_fp()


  def get_step_frames(self, rate):
    return [int(round(t * rate)) for _, _, t, _ in self.steps]