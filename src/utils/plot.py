import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


metric = 'accuracy'
data_set = 'train'
base_path = '../../logs/' + metric + '/' + data_set

df = pd.DataFrame({'epochs' : list(range(12))})

for filename in os.listdir(base_path):
	file_path = os.path.join(base_path,filename)
	tmp_df = pd.read_csv(file_path, usecols = [2], header = 0, names = [filename[:-4]])
	df = pd.concat([df, tmp_df], axis=1)

df.plot(x = 'epochs')
plt.show()
#TODO: Add xticks, yticks, colors, grid

print(df.head())