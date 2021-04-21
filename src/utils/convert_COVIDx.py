import os
import sys

from definitions import DATA_DIR
from shutil import copy

COVIDx_path = os.path.join(DATA_DIR, 'COVIDx8')

print(COVIDx_path)

test_split = os.path.join(COVIDx8_path, 'test_split.txt')
train_split = os.path.join(COVIDx8_path, 'train_split.txt')
train_path = os.path.join(COVIDx8_path, 'train')
test_path = os.path.join(COVIDx8_path, 'test')

for path in (train_path, test_path):
    for label in ('positive', 'negative'):
        try:
            os.mkdir(os.path.join(path, label))
        except FileExistsError:
            pass

with open(test_split, 'r') as test_file:
    for line in test_file:
        line_splited = line.split(' ')
        img_path = os.path.join(test_path, line_splited[1])
        label = line_splited[2]
        moved = move(img_path, os.path.join(test_path, label))
        print(img_path, " -> ", moved)
