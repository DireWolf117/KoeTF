import tensorflow as tf
import numpy as np
import datautils as utils
import random

from random import randrange
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

def autoencoder():
  initializer = tf.keras.initializers.GlorotNormal()

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='elu', input_shape=(5632,), kernel_initializer=initializer),
    tf.keras.layers.Dense(256, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(64, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(16, activation='tanh', kernel_initializer=initializer),
    tf.keras.layers.Dense(64, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(256, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(1024, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(5632, activation='sigmoid', kernel_initializer=initializer),
  ])

  return model

def decoder(original_ae):
  model = tf.keras.Sequential(name="Decoder")

  for i in range(4):
    model.add(original_ae.layers[i + 4])
    if not np.equal(model.layers[i].get_weights, original_ae.layers[i+4].get_weights):
      print("ERROR: Weight of Decoder not the same!")
  model.build(input_shape=[None,16])
  model.summary()

  return model

def encoder(original_ae):
  model = tf.keras.Sequential(name="Encoder")
  for i in range(4):
    model.add(original_ae.layers[i])
    if not np.equal(model.layers[i].get_weights, original_ae.layers[i].get_weights):
      print("ERROR: Weight of Encoder not the same!")
  model.summary()

  return model


def lstm():
  model = tf.keras.Sequential([
    tf.keras.layers.LSTM(500, input_dim=88, return_sequences=True),
    tf.keras.layers.LSTM(500, return_sequences=True),
    tf.keras.layers.Dense(88, activation='sigmoid'),
  ])

  return model

def lstm_generate(model, input, timesteps, temp = 1.0):
  y = np.zeros((1, timesteps+1, len(input)), float)
  print("From lstm_generate(model, input, timesteps), Data Matrix of size: " + str(y.shape))
  y[0, 0] = input
  for i in range(timesteps):
    x = np.array(model(y)[0, i:i+1])
  #  done = False
  #  for e in range(len(x[0])):
  #    if 0.2 < x[0][e]:
  #      x[0][e] = 1.0
  #    else:
  #      x[0][e] = 0.0
  #    if (random.random() / temp) < x[0][e]:
  #      x[0][e] = 1.0
  #    else:
  #      x[0][e] = 0.0
    #if i > 8:
    #  for f in range(8):
    #    if not np.array_equal(x[0], y[0][i - f]):
    #      break
    #    if f == 7:
    #      print("Same")
    #      z = random.randint(40, 60)
    #      x[0][z] = 1.0
    y[0][i+1] = x
  return y

class test_and_generate_callback(tf.keras.callbacks.Callback):
  def __init__(self, nn_type):
    self.nn_type = nn_type

  def on_epoch_end(self, epoch, logs=None):
    if self.nn_type == "LSTM":
      print('\n')
      #Evaluate on testdata
      print("-------- Evaluating on testset --------")
      x, y = utils.lstm_dataset("data/Dataset/validation/", 16*16, add_labels=True, num_files = 1)
      x = x[0:, :1]
      results = self.model.evaluate(x, y, batch_size=1)
      print(results)
      loss_file = open("logs/test_loss.txt", "a")
      accuracy_file = open("logs/test_accuracy.txt", "a")
      loss_file.write(str(results[0]) + "\n")
      accuracy_file.write(str(results[1]) + "\n")
      loss_file.close()
      accuracy_file.close()

      #Generate a song
      y = lstm_generate(self.model, x[0][0], 96, temp = 4.0)
      utils.MHE_to_txt(y, 0.5, epoch)
      print("\n")
    else: 
      print('\n')
      #Evaluate on testdata
      print("-------- Evaluating on testset --------")
      #x = utils.data_from_npz_MHE('data/numpy_arrays/1_5632.npy').astype(np.float64)
      x = utils.read_data_to_mat_MHE("data/Dataset/validation/", 1000, 4*16, False)
      results = self.model.evaluate(x, x, batch_size=1)
      print(results)
      loss_file = open("logs/test_loss.txt", "a")
      accuracy_file = open("logs/test_accuracy.txt", "a")
      loss_file.write(str(results[0]) + "\n")
      accuracy_file.write(str(results[1]) + "\n")
      loss_file.close()
      accuracy_file.close()

      #Generate a song
      y = self.model(x[100:101])
      utils.MHE_to_txt(y, 0.5, epoch)
      print("\n")
 


