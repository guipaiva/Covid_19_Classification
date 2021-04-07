import os
from shutil import copyfile

test_split = 'test_split.txt'
train_split = 'train_split.txt'
train_path = 'data/train'
test_path = 'data/test'


try:
    #Treino
    os.mkdir(os.path.join(train_path, 'positive'))
    os.mkdir(os.path.join(train_path, 'negative'))  

    #Teste
    os.mkdir(os.path.join(test_path, 'positive'))
    os.mkdir(os.path.join(test_path, 'negative'))
except:
    pass


with open(test_split, 'r') as test_file:
    for line in test_file:
        pass
        