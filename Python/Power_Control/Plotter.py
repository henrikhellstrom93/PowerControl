# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:49:44 2021

@author: Henke
"""
import matplotlib.pyplot as plt

#--Load acc_history from files
#filenames = ["0noise-10hidden", "0.1noise-10hidden-trunc", "0.1noise-10hidden-henrik"]
#filenames = ["0.1noise-100hidden-xiaowen", "0.1noise-100hidden-henrik"]
#filenames = ["0noise-10hidden-xiaowen", "1noise-10hidden-xiaowen", "1noise-10hidden-xiaowen-1rtx"]
#filenames = ["0.1noise-10hidden-xiaowen-10rtx", "0.1noise-10hidden-henrik-10rtx"]
#filenames = ["1noise-10hidden-henrik-0rtx-50budget", "1noise-10hidden-henrik-1rtx-50budget", "1noise-10hidden-henrik-3rtx-50budget", "1noise-10hidden-henrik-7rtx-50budget"]
filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-1rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-7rtx-100budget", "1noise-10hidden-henrik-15rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
#filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
acc_histories = []

for filename in filenames:
    acc_history = []
    with open("./data/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            acc_history.append(float(line))
    acc_histories.append(acc_history)
    
#Find max length
max_len = 1
for acc_history in acc_histories:
    if len(acc_history) > max_len:
        max_len = len(acc_history)
        
#--Plot acc_histories
start = 0
end = -1
fig = plt.figure()
i = 0
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for acc_history in acc_histories:
    plt.plot(range(len(acc_history)), acc_history, colors[i], label=filenames[i])
    dashed_line_length = max_len-len(acc_history)
    plt.hlines(acc_history[len(acc_history)-1], len(acc_history)-1, max_len, colors[i], linestyles='dashed')
    #plt.plot(range(dashed_line_length), [acc_history[len(acc_history)-1]]*dashed_line_length, colors[i] + "--", xmin=len(acc_history))
    i = i + 1
plt.legend()
compareplot_name = "100budget-comparison"
plt.savefig("./plots/"+compareplot_name+".png", format='png')