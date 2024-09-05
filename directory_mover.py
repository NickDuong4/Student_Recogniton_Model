import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from sklearn.model_selection import train_test_split

import shutil
from shutil import unpack_archive
from subprocess import check_output


###file is to move files from main folder of persons into training/validation/testing sets
fullPath = os.path.abspath("./lfw_funneled")

#load list of names in LFW_FUNNELED 
names = pd.read_csv(os.path.abspath("./csvs/lfw_allnames.csv"))

image_paths = names.loc[names.index.repeat(names['images'])]
image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
image_paths.drop(["images"], axis=1, inplace=True)



#take people between 443 and 633 of images (group members number of photos, not the best way of doing this)
image_paths = image_paths.groupby("name").filter(lambda x: ((len(x) > 443) & (len(x) < 633)))
class_number = len(pd.unique(image_paths['name']))
print(class_number)



#make train/validation/test splits
train_ds, test_ds = train_test_split(image_paths, test_size=0.1)
train_ds = train_ds.reset_index().drop("index", axis=1)
test_ds = test_ds.reset_index().drop("index", axis=1)

train_ds, valid_ds = train_test_split(train_ds, test_size=0.15)

print(len(set(train_ds.name).intersection(set(test_ds.name))))
print(len(set(test_ds.name) - set(train_ds.name)))

print(os.path.abspath("./lfw_funneled"))


def directory_mover(data,dir_name):
    co = 0
    for image in data.image_path:
        # create top directory
        if not os.path.exists(os.path.abspath("./"+dir_name)):
            shutil.os.mkdir(os.path.abspath('./'+dir_name))
        
        data_type = data[data['image_path'] == image]['name']
        data_type = str(list(data_type)[0])
        
        if not os.path.exists(os.path.join('./'+dir_name,data_type)):
            shutil.os.mkdir(os.path.join('./'+dir_name,data_type))
            
        path_from = os.path.join('./lfw_funneled/'+image)
        path_to = os.path.join('./'+dir_name,data_type)
        # print(path_to)
        
        shutil.copy(path_from, path_to)
        # print('Moved {} to {}'.format(image,path_to))
        co += 1
        
    print('Moved {} images to {} folder.'.format(co,dir_name))

#makes folders if they don't exist
if os.path.exists(os.path.abspath("./train_ds/")):
            shutil.rmtree(os.path.abspath("./train_ds/"))
if os.path.exists(os.path.abspath("./valid_ds/")):
            shutil.rmtree(os.path.abspath("./valid_ds/"))      
if os.path.exists(os.path.abspath("./test_ds/")):
            shutil.rmtree(os.path.abspath("./test_ds/"))


#call directory mover for all 3 datasets
directory_mover(train_ds, "train_ds/")
directory_mover(test_ds, "test_ds/")
directory_mover(valid_ds, "valid_ds/")

