# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:33:49 2022

@author: inalt
"""

# import libraries

from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical
from keras.models import load_model

from sklearn.model_selection import train_test_split

#  load model

skin_df = pd.read_csv("HAM10000_metadata.csv")


# Data Cleaning

skin_df.isnull().sum()

# fill in the blanks according to their mean

skin_df["age"].fillna((skin_df["age"].mean()), inplace = True)

# check the presence of null values again

skin_df.isnull().sum()

# EDA

# visualize

skin_df["dx"].value_counts().plot(kind = "bar")
skin_df['dx_type'].value_counts().plot(kind = 'bar')
skin_df["localization"].value_counts().plot(kind = "bar")
skin_df["age"].hist(bins = 50)
skin_df["sex"].value_counts().plot(kind = "bar")
sns.scatterplot("age", "dx_id", data = skin_df)

skin_df.head()
skin_df.info()
skin_df.describe()

plt.show()

# preprocess

data_folder_name = "HAM10000_images_part_1/"
ext = ".jpg" # extantion

# data_folder_name + image_id[i] + ext => ex: "HAM10000_images_part_1\ISIC__0027622.jpg"
skin_df["path"] = [data_folder_name + i + ext for i in skin_df["image_id"]]
skin_df["image"] = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

plt.imshow(skin_df["image"][300])

skin_df["dx_id"] = pd.Categorical(skin_df["dx"]).codes # object to int
"""

akiec = 0
bcc = 1
bkl = 2
df = 3
mel = 4
nv = 5
vasc = 6

"""

skin_df.to_pickle("skin_df.pkl") # save as pickle
# read pickle

skin_df = pd.read_pickle("skin_df.pkl")
# checking the image size distribution

skin_df["image"].map(lambda x: x.shape).value_counts() # (75, 100, 3)    10015
                                                       # Name: image, dtype: int64
      
# standardization

x = np.asarray(skin_df["image"].tolist())
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

# one hot encoding

y = to_categorical(skin_df["dx_id"], num_classes = 7)

# train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Set the CNN Model

# my CNN architechture is [[Conv2D => relu] * 2  -> MaxPool2D -> Dropout] * 2 => Flatten => Dense

input_shape = (75,100,3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25)) # prevents over-fitting

model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = "softmax"))

model.summary()

# define the optimizer

optimizer = Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 25
batch_size = 10

# verbose = the outputs of the model are opened
# shuffle = mix
history = model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, verbose = 1, shuffle = True) 

model.save("my_model1.h5")

# loss: 0.3708 - accuracy: 0.8684
