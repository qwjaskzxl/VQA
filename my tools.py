import urllib, requests
import zipfile
import os, sys, shutil

def delete_dir():
    shutil.rmtree('dataset/Images/train2014')

def download():
    url = 'http://images.cocodataset.org/zips/test2015.zip'
    urllib.request.urlretrieve(url, "dataset/Images/test2015.zip")
    # url = 'http://images.cocodataset.org/zips/val2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/val2014.zip")
    # url = 'http://images.cocodataset.org/zips/train2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/train2014.zip")

def unzip():
    # with zipfile.ZipFile('dataset/Images/train2014.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    # with zipfile.ZipFile('dataset/Images/val2014.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    # with zipfile.ZipFile('dataset/Images/test2015.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    with zipfile.ZipFile('dataset/Annotations_Train_mscoco.zip', 'r') as zf, \
         zipfile.ZipFile('dataset/Questions_Train_mscoco.zip', 'r') as zf2:
        zf.extractall(path='dataset')
        zf2.extractall(path='dataset')

# download()
# unzip()

from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter()
for e in range(100):
    writer.add_scalar('scaler/test', np.random.rand(), e)
    writer.add_scalars('scaler/scalers_test', {'xsinx':e*np.sin(e), 'xcosx':e*np.cos(e)}, e)
writer.close()