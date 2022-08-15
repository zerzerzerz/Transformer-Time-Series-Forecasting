import pandas as pd
from glob import glob
from utils.utils import load_json, save_json





# for p in glob('data/*.csv'):
#     name = p.split('/')[-1][0:-4]
#     shape = pd.read_csv(p).shape
#     length, channel = shape
#     channel -= 1
#     print('Length of {:<20} is {:<6}, channel is {:<3}'.format(name,length,channel))