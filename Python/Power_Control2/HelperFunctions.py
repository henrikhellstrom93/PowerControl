# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:13:18 2021

@author: Henke
"""
import numpy as np

def vectorifyUpdates(weight_updates, single_model):
    #If we got multiple neural networks
    if single_model == False:
        num_devices = len(weight_updates)
        num_layers = int(len(weight_updates[0])/2)
        num_weights_layer1 = len(weight_updates[0][0])*len(weight_updates[0][0][0])
        num_bias_layer1 = len(weight_updates[0][1])
        num_weights_layer2 = len(weight_updates[0][2])*len(weight_updates[0][2][0])
        num_bias_layer2 = len(weight_updates[0][3])
        vector_length = num_weights_layer1+num_bias_layer1+num_weights_layer2+num_bias_layer2
        ret_vectors = []
        for i in range(num_devices):
            ret_vector = np.zeros((vector_length,1))
            pointer = 0
            for j in range(num_layers*2):
                #If j%2 == 0, then we are looking at weights
                if j % 2 == 0:
                    weight_vector = np.concatenate(weight_updates[i][j])
                    num_weights = len(weight_vector)
                    ret_vector[pointer:pointer+num_weights] = weight_vector.reshape((num_weights,1))
                    pointer = pointer + num_weights
                #If j%2 == 1, then we are looking at biases
                elif j % 2 == 1:
                    bias_vector = weight_updates[i][j]
                    num_biases = len(bias_vector)
                    ret_vector[pointer:pointer+num_biases] = bias_vector.reshape((num_biases,1))
                    pointer = pointer + num_biases
            ret_vectors.append(ret_vector)
        return ret_vectors
    #If we only got one model
    else:
        num_layers = int(len(weight_updates)/2)
        num_weights_layer1 = len(weight_updates[0])*len(weight_updates[0][0])
        num_bias_layer1 = len(weight_updates[1])
        num_weights_layer2 = len(weight_updates[2])*len(weight_updates[2][0])
        num_bias_layer2 = len(weight_updates[3])
        vector_length = num_weights_layer1+num_bias_layer1+num_weights_layer2+num_bias_layer2
        ret_vector = np.zeros((vector_length,1))
        pointer = 0
        for j in range(num_layers*2):
            #If j%2 == 0, then we are looking at weights
            if j % 2 == 0:
                weight_vector = np.concatenate(weight_updates[j])
                num_weights = len(weight_vector)
                ret_vector[pointer:pointer+num_weights] = weight_vector.reshape((num_weights,1))
                pointer = pointer + num_weights
            #If j%2 == 1, then we are looking at biases
            elif j % 2 == 1:
                bias_vector = weight_updates[j]
                num_biases = len(bias_vector)
                ret_vector[pointer:pointer+num_biases] = bias_vector.reshape((num_biases,1))
                pointer = pointer + num_biases
        return ret_vector

def calculateMeanVector(vector_list):
    num_vectors = len(vector_list)
    ret_vector = np.zeros(vector_list[0].shape)
    for i in range(num_vectors):
        ret_vector = ret_vector + vector_list[i]
    ret_vector = ret_vector/num_vectors
    return ret_vector