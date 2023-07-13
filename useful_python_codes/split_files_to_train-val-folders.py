import os
import shutil

path = 'images/'
train_path = 'images/train'
val_path = 'images/val'

files = [f for f in os.listdir(path) if f.endswith('.jpg')]  # you can change to .png or .txt

n = len(files)
train_n = int(n * 0.8)

print(n)

for i in range(train_n):
    shutil.move(os.path.join(path, files[i]), train_path)

for i in range(train_n, n):
    shutil.move(os.path.join(path, files[i]), val_path)
