import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot(base_path):
	df = pd.DataFrame({'epochs' : list(range(12))})
	for filename in os.listdir(base_path):
		file_path = os.path.join(base_path,filename)
		tmp_df = pd.read_csv(file_path, usecols = [2], header = 0, names = [filename[:-4]])
		df = pd.concat([df, tmp_df], axis=1)

	_, ax = plt.subplots()

	for column in df.columns[1:]:
		ax.plot(df['epochs'], df[column], label = column)

	ax.legend(loc = 'best')
	ax.grid(True)
	ax.set_xticks(range(12))
	ax.set_xlabel("Epochs")
	ax.set_ylabel(metric.capitalize())
	plt.savefig(logs_path + '/' + subset+ '_' + metric)

metrics = ['accuracy','loss']
subsets = ['train','test']

logs_path = os.path.abspath(os.path.dirname(__file__) + "/../../logs")

for metric in metrics:
	for subset in subsets:
		plot(os.path.join(logs_path, metric, subset))