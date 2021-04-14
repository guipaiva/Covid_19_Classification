import os
from shutil import move

test_split = 'data/test_split.txt'
train_split = 'data/train_split.txt'
train_path = 'data/train'
test_path = 'data/test'

for path in (train_path, test_path):
    for label in ('positive','negative'):
        try:
            os.mkdir(os.path.join(path, label))
        except FileExistsError:
            pass

with open(test_split, 'r') as test_file:
    for line in test_file:
        line_splited = line.split(' ')
        img_path = os.path.join(test_path,line_splited[1])
        label = line_splited[2]
        moved = move(img_path, os.path.join(test_path, label))
        print(img_path, " -> ", moved)


        