import csv
import os

def write_raw(data, log_dir, model_name):
		os.makedirs(log_dir, exist_ok= True)
		fname = os.path.join(log_dir, model_name + '.csv')
		with open(fname,'w') as file:
			writer = csv.writer(file)
			writer.writerow([str(key) for key in data.keys()])
			zip_data = zip(data.values())
			writer.writerows(zip_data)