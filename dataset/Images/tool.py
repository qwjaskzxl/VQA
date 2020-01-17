# import glob
# len(glob.glob('./dataset/Images/val2014/*.jpg'))*3#或者指定文件下个数

import os
count = 0
name = []
f=open('rest.txt','r')
names = [a.strip() for a in f.readlines()]

import shutil
for a in names:
    shutil.copyfile('./val2014/%s'%a, './a/%s'%a)

# =============================================================================
# for root,dirs,files in os.walk('val2014'):    #遍历统计
#     for each in names:
#         
#         count += 1   #统计文件夹下文件个数
#              f.write(each+'\n')
# print( count,121512//3  ) 
# =============================================================================
import zipfile
with zipfile.ZipFile('dataset/Images/test2015.zip', 'r') as zf:
    zf.extractall(path='dataset/Images/test2015')

