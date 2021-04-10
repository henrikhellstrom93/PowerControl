# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:09:19 2021

@author: Henke
"""
#--Simulation settings--
num_rounds = 20 # Number of communication rounds
num_devices = 10 # Number of devices
SNR = 1/2 #Signal-to-Noise ratio in linear scale (-1 means no noise)
bs = 50 # Batch size for local training at devices
ep = 2 # Number of local epochs before communication round
filename = "1noise-10hidden-xiaowen-2rtx"
# powercontrol = "henrik"
powercontrol = "xiaowen"

#--Load MNIST dataset--
import tensorflow as tf
import time
import numpy as np
from Power_Control import PowerControl

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

num_samples = len(x_train)

#Split dataset into shards
samples_per_shard = int(num_samples/num_devices)
x_train_shards = []
y_train_shards = []
for i in range(num_devices):
    x_train_shard = x_train[i*samples_per_shard:(i+1)*samples_per_shard]
    x_train_shards.append(x_train_shard)
    y_train_shard = y_train[i*samples_per_shard:(i+1)*samples_per_shard]
    y_train_shards.append(y_train_shard)

print("Dataset loaded.")

#--Set up DNN models--
model_template = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model_list = []

#Create user device models and global model
global_model = tf.keras.models.clone_model(model_template)
for i in range(num_devices):
    model_list.append(tf.keras.models.clone_model(model_template))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Compile all models to initiate weights
for model in model_list:
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#Broadcast global model to all user devices
global_weights = global_model.get_weights()
for model in model_list:
    model.set_weights(global_weights)
    
#--Perform FL--
sigma_w = 1;
rtx = 2;
pc = PowerControl(num_devices, sigma_w, rtx)
acc_history = []
for r in range(num_rounds):
    print("Communication round " + str(r+1) + "/" + str(num_rounds))
    start = time.time()
    #Train using local dataset
    for d in range(num_devices): #TODO: Parallelize
        model_list[d].fit(x_train_shards[d], y_train_shards[d], batch_size=bs, epochs=ep, verbose=0)
    
    #Calculate weight updates
    new_weights = []
    weight_updates = []
    for d in range(num_devices):
        new_weights.append(model_list[d].get_weights())
        weight_updates.append(new_weights[d]) #Just to initiate shape
        for l in range(len(weight_updates[d])):
            weight_updates[d][l] = new_weights[d][l] - global_weights[l]
            
    #Power Control to get sum
    sum_update = weight_updates[0] #Just to initiate shape
    num_layers = len(sum_update)
    for l in range(num_layers):
        #Even layers contain weights
        if l%2 == 0:
            layer_height = len(weight_updates[0][l])
            layer_width = len(weight_updates[0][l][0])
            #We get one row of the weight matrix at a time
            for r in range(layer_height):
                row_matrix = np.zeros((num_devices, layer_width))
                for d in range(num_devices):
                    row = weight_updates[d][l][r]
                    row_matrix[d,:] = row
                row_sum = pc.estLayer(row_matrix, powercontrol)
                sum_update[l][r] = row_sum[:].reshape(layer_width,)
        else:
            layer_width = len(weight_updates[0][l])
            row_matrix = np.zeros((num_devices, layer_width))
            for d in range(num_devices):
                row = weight_updates[d][l]
                row_matrix[d,:] = row
            row_sum = pc.estLayer(row_matrix, powercontrol)
            sum_update[l] = row_sum[:].reshape(layer_width,)
                    
    #Average at server
    average_update = sum_update
    for l in range(num_layers):
        average_update[l] = average_update[l]/num_devices
    
    #Update global model
    new_global = global_weights #Just to initiate shape
    for l in range(len(new_global)):
        new_global[l] = global_weights[l] + average_update[l]
    #Broadcast global model to user devices
    for model in model_list:
        model.set_weights(new_global)
    global_weights = new_global
    #Logging
    acc_history.append(model_list[0].evaluate(x_test, y_test, verbose=0)[1])
    print("Current test dataset accuracy: ", acc_history[-1])
    print(str(int(time.time()-start)) + " seconds elapsed\n")

#--Plot accuracy--
import matplotlib.pyplot as plt
start = 0
end = -1
fig = plt.figure()
plt.plot(range(len(acc_history[start:end])), acc_history[start:end])
plt.savefig("./plots/"+filename+".png", format='png')

#--Store acc_history in file--
with open("./data/"+filename+".txt", "w") as filehandle:
    for item in acc_history:
        filehandle.write("%s\n" % item)