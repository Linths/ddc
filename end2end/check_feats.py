import pickle
import numpy as np

path = "data/feats/fraxtil/mel80hop441/Fraxtil_sArrowArrangements_BadKetchup.pkl"
with open(path, 'rb') as f:
      data = pickle.load(f)
print(data.shape)