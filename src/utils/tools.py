import csv
import os

def write_raw(data, log_dir, model_name):
		os.makedirs(log_dir, exist_ok= True)
		fname = os.path.join(log_dir, model_name + '.csv')
		with open(fname,'w') as file:
			writer = csv.writer(file)
			writer.writerow(('train_acc','train_loss','validation_acc','validation_loss'))
			zip_data = zip(data['loss'], data['accuracy'], data['val_accuracy'], data['val_loss'])
			writer.writerows(zip_data)