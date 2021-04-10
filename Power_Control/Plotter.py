# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:49:44 2021

@author: Henke
"""
import matplotlib.pyplot as plt

#--Load acc_history from files
#filenames = ["0noise-10hidden", "0.1noise-10hidden-trunc", "0.1noise-10hidden-henrik"]
#filenames = ["0.1noise-100hidden-xiaowen", "0.1noise-100hidden-henrik"]
filenames = ["0noise-10hidden-xiaowen", "1noise-10hidden-xiaowen", "1noise-10hidden-xiaowen-1rtx"]
acc_histories = []

for filename in filenames:
    acc_history = []
    with open("./data/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            acc_history.append(float(line))
    acc_histories.append(acc_history)
    
#--Plot acc_histories
start = 0
end = -1
fig = plt.figure()
i = 0
for acc_history in acc_histories:
    plt.plot(range(len(acc_history[start:end])), acc_history[start:end], label=filenames[i])
    i = i + 1
plt.legend()
plt.savefig("./plots/"+filename+".png", format='png')