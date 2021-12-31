import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import sys
import tensorflow as tf
import tensorboard
import numpy as np
import datautils as utils
import models as mod
import gui
import shutil

from tensorflow import keras
from datetime import datetime
from packaging import version
from tensorflow.python.client import device_lib


model = tf.keras.models.load_model(filepath="Models/working_models/ae_5632.ckpt")


#model = mod.autoencoder()
#adam = tf.keras.optimizers.Adam(learning_rate=0.001)
#model.compile(optimizer=adam, loss=tf.keras.losses.MeanSquaredError(), metrics=['binary_accuracy'])

print(model.summary())
#model.layers[1].rate = 0.0
#
gui.gen_gui(model)
#gui.show_validation_loss()

goon = ''
while goon != 'n' and goon != 'Y' and goon != 'y' and goon != 'N':
    goon = input("Do you want to continue with execution? All saved files will be overwritten. Y/N ")
    if goon == 'n' or goon == 'N':
        sys.exit()


#Checkpoints
checkpoint_path = "Models/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, verbose=1, save_best_only=False, save_freq=100000)
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, )
model.save(checkpoint_path.format(epoch=0))
test_cp = mod.test_and_generate_callback("LSTM")


#Data
train_data, train_labels = utils.lstm_dataset("data/Dataset/train", 16*16, True)
train_data = train_data[0:, :1]
print(train_data[0])
print(train_data[1])
print(train_data[2])
print(train_data[3])
print(train_data[4])
print(train_data[5])

#train_data = utils.read_data_to_mat_MHE("data/Dataset/train", 1, 4*16, False)
#train_labels = train_data
print("Training Data after Shuffle and Duplication: " + str(train_data.shape))


#Training
cp_callback.filepath = "Models/cp-{epoch:04d}.ckpt" 
tensorboard_callback.log_dir = "logs/fit/" + datetime.now().strftime("%m/%d-%H%M%S")
model.fit(train_data, train_labels, batch_size=10, epochs=1000, callbacks=[cp_callback, tensorboard_callback, test_cp])


